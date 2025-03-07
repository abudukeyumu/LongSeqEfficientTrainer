import torch
import datasets
from transformers import Trainer, AutoModel, AutoTokenizer, TrainingArguments
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
# torch.autograd.set_detect_anomaly(True)
import argparse
from datasets import disable_caching

disable_caching()


def extend_position_embeddings(model, new_max_length):
    """
    扩展 BERT 模型的位置编码矩阵，并更新 position_ids 和 token_type_ids buffer。

    Args:
        model:  预训练的 BERT 模型 (transformers.BertModel)
        new_max_length:  新的最大序列长度
    """
    if new_max_length <= model.config.max_position_embeddings:
        print(f"新的最大长度 {new_max_length} 没有超过当前的 {model.config.max_position_embeddings}，无需扩展。")
        return

    old_position_embeddings = model.embeddings.position_embeddings.weight.data
    old_max_length = model.config.max_position_embeddings
    embedding_dim = old_position_embeddings.size(1)

    # 创建新的位置编码矩阵
    new_position_embeddings = nn.Embedding(new_max_length, embedding_dim).weight.data
    torch.nn.init.normal_(new_position_embeddings, mean=0.0, std=model.config.initializer_range)

    # 复制旧的位置编码
    n = min(old_max_length, new_max_length)
    new_position_embeddings[:n, :] = old_position_embeddings[:n, :]

    # 替换模型中的位置编码权重
    model.embeddings.position_embeddings = nn.Embedding.from_pretrained(new_position_embeddings)

    # **关键修改：重新注册 position_ids 和 token_type_ids buffer**
    model.embeddings.register_buffer(
        "position_ids", torch.arange(new_max_length).expand((1, -1)), persistent=False
    )
    model.embeddings.register_buffer(
        "token_type_ids", torch.zeros((1, new_max_length), dtype=torch.long), persistent=False
    )


    # 更新模型配置
    model.config.max_position_embeddings = new_max_length
    model.embeddings.position_ids.max_len = new_max_length # 确保 position_ids 的 max_len 也更新 (某些模型结构可能需要)


    print(f"位置编码已扩展到 {new_max_length}，position_ids 和 token_type_ids buffer 已更新。")




def format(item):
    # 合并历史对话，奇偶行区分 User 和 Agent
    anchor = ""
    
    for i, line in enumerate(item["history"][:-1]):
        if i % 2 != 0:
            anchor += "Agent: " + line
        else:
            anchor += "User: " + line

        if i < len(item["history"][:-1]) - 1:
            anchor += "\n"

    positive = item["gold_document"]
    negatives = item["document_list"]
    

    del negatives[int(item["local_did"][-1]) - 1]


    return {
        "anchor": anchor,
        "positive": positive,
        "negative": negatives  # 依然保留一个列表，以便多负例
    }



def load_dataset(path):
    dataset = datasets.load_from_disk(path)
    
    dataset = dataset.filter(lambda x: len(x['document_list'])>8,num_proc=32)
    dataset = dataset.shuffle(seed=42,keep_in_memory=True)
    # dataset = dataset.select()
    
    
    dataset = dataset.map(format, num_proc=32, remove_columns=dataset.column_names)

    
    return dataset

# ---------------------------
# 3) collate_fn: 将批量数据 token 化并返回字典
#    若负例是多个，会展平后再记录每条的负例数量，供 compute_loss 重排用
# ---------------------------
def collate_fn(batch, query_tokenizer,context_tokenizer,max_length):
    anchors = [item["anchor"] for item in batch]
    positives = [item["positive"] for item in batch]
    negatives_list = [item["negative"] for item in batch]

    anchor_input = query_tokenizer(
        anchors,
        padding=True,
        truncation=True,     # <-- Add
        max_length=max_length,      # <-- Add
        return_tensors="pt",
    )
    positive_input = context_tokenizer(
        positives,
        padding=True,
        truncation=True,     # <-- Add
        max_length=max_length,      # <-- Add
        return_tensors="pt",
    )

    flattened_negatives = []
    for neg_docs in negatives_list:
        flattened_negatives.extend(neg_docs)

    
    negative_input = context_tokenizer(
        flattened_negatives,
        padding=True,
        truncation=True,     # <-- Add
        max_length=max_length,      # <-- Add
        return_tensors="pt",
    )

    return {
        "anchor_input": anchor_input,
        "positive_input": positive_input,
        "negative_input": negative_input,
    }


class BiEncoderTrainer(Trainer):


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        anchor_input = inputs["anchor_input"]
        positive_input = inputs["positive_input"]
        negative_input = inputs["negative_input"]

        batch_size=1
        # 编码 anchor 和 positive
        anchor_embeds_list = []
        num_anchors = anchor_input["input_ids"].size(0)
        for i in range(0, num_anchors, batch_size):
            batch_anchor_input = {
                key: value[i:i+batch_size] for key, value in anchor_input.items()
            }
            anchor_outputs = model(**batch_anchor_input)
            anchor_embeds = anchor_outputs.last_hidden_state[:, 0]
            anchor_embeds_list.append(anchor_embeds)

        # 拼接所有 anchor_embeds
        anchor_embeds = torch.cat(anchor_embeds_list, dim=0)

        # 分批处理 positive_input
        positive_embeds_list = []
        num_positives = positive_input["input_ids"].size(0)
        for i in range(0, num_positives, batch_size):
            batch_positive_input = {
                key: value[i:i+batch_size] for key, value in positive_input.items()
            }
            positive_outputs = model(**batch_positive_input)
            positive_embeds = positive_outputs.last_hidden_state[:, 0]
            positive_embeds_list.append(positive_embeds)

        # 拼接所有 positive_embeds
        positive_embeds = torch.cat(positive_embeds_list, dim=0)
        
        # negative_outputs = model(**negative_input) 
        # negative_embeds = negative_outputs.last_hidden_state[:, 0]
        
        negative_embeds_list = []
        num_negatives = negative_input["input_ids"].size(0)
        for i in range(0, num_negatives, batch_size):
            batch_negative_input = {
                key: value[i:i+batch_size] for key, value in negative_input.items()
            }
            negative_outputs = model(**batch_negative_input)
            negative_embeds = negative_outputs.last_hidden_state[:, 0]
            negative_embeds_list.append(negative_embeds)

        # 拼接所有 negative_embeds
        negative_embeds = torch.cat(negative_embeds_list, dim=0)

        anchors = anchor_embeds  # (batch_size, embedding_dim)
        
        # 拼接正负例的嵌入
        candidates = torch.cat([positive_embeds, negative_embeds], dim=0)  # (batch_size * (1 + num_negatives), embedding_dim)

        # 注意下面这个乘的系数可以增大loss，从而保证grad_norm不会消失，否则在fp16的情况下可能会变成NAN，根据具体数据集具体调整就好。
        # 计算余弦相似度
        # F.cosine_similarity expects the input tensors to have shape (batch_size, embedding_dim)
        scores = F.cosine_similarity(anchors.unsqueeze(1), candidates.unsqueeze(0), dim=2) *20
        # (batch_size, batch_size * (1 + num_negatives))
        
        
        # anchor[i] should be most similar to candidates[i], as that is the paired positive,
        # so the label for anchor[i] is i
        range_labels = torch.arange(0, scores.size(0), device=scores.device)
        loss_func = nn.CrossEntropyLoss()
     
        return loss_func(scores, range_labels)

       
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_mode',default='synthethisqa')
    parser.add_argument('--model_dir')
    parser.add_argument('--save_name')
    return parser.parse_args()


if __name__=='__main__':

    args = parse_args()

    model_path= args.model_dir

    max_length=8192
    # 加载模型和 tokenizer
    
    query_tokenizer= AutoTokenizer.from_pretrained(model_path,truncation_side='left')
    context_tokenizer = AutoTokenizer.from_pretrained(model_path)

        
    if args.data_mode=='synthethisqa':
        #dataset = datasets.load_from_disk(synthethisqa_path)
        def convert_negative_to_list(example):
            example['negative'] = [example['negative']]
            return example
        from datasets import load_dataset as hf_load_dataset
        dataset_dict =hf_load_dataset("parquet", data_files="/mnt/abu/data/synthethisQA-retrieval/*.parquet")
        dataset_old = dataset_dict["train"]
        dataset = dataset_old.map(convert_negative_to_list)
    print(len(dataset))
    print(dataset[0]['anchor'])

    model = AutoModel.from_pretrained(model_path)
    extend_position_embeddings(model,max_length)
    # 训练参数
    training_args = TrainingArguments(
        output_dir=f"/mnt/abu/evaluation/xiangmu/bge/{args.save_name}-{args.data_mode}",
        overwrite_output_dir=True,
        logging_dir="./logs",
        logging_steps=1,
        save_strategy='epoch',
        save_total_limit=2,
        # max_steps=900,
        num_train_epochs=3,
        warmup_ratio=0.05,
        learning_rate=3e-5,
        weight_decay=0.01,
        gradient_accumulation_steps=16,
        per_device_train_batch_size=4,
        dataloader_num_workers=16,
        fp16=True,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        # gradient_checkpointing=True,
    )

    # 初始化 Trainer
    trainer = BiEncoderTrainer(
        model=model,
        args=training_args,
        tokenizer=query_tokenizer,
        train_dataset=dataset,
        data_collator=lambda batch: collate_fn(batch, query_tokenizer,context_tokenizer,max_length)
    )
    # ddd=trainer.get_train_dataloader()
    # for d in ddd:
    #     import pdb
    #     pdb.set_trace()
    trainer.train()
