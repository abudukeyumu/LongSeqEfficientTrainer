import torch
import datasets
from transformers import Trainer, AutoModel, AutoTokenizer, TrainingArguments
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
# torch.autograd.set_detect_anomaly(True)
import argparse
from datasets import disable_caching
from torch.utils.checkpoint import get_device_states, set_device_states

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


class RandContext:
    """
    Random-state context manager class. Reference: https://github.com/luyug/GradCache.

    This class will back up the pytorch's random state during initialization. Then when the context is activated,
    the class will set up the random state with the backed-up one.
    """
    def __init__(self, *tensors) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self) -> None:
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None
        
class BiEncoderTrainerWithGradCache(Trainer):
    def __init__(self, *args, mini_batch_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.mini_batch_size = mini_batch_size
        self.grad_scale = 20.0  # 相似度缩放因子
        self.cache = None   #用来保存梯度
        self.random_states = None  #用来保存随机状态确保第二次前向传播时能连接到第一次
    
    def encode_minibatch(self, model, input_dict, start_idx, end_idx, with_grad=True, rand_state=None):
        mini_batch_input = {k: v[start_idx:end_idx] for k, v in input_dict.items()}
        tensors = list(mini_batch_input.values())
        
        # 如果没有提供随机状态且需要复制随机状态，则创建一个新的随机状态
        if rand_state is None:
            rand_state = RandContext(*tensors)
            need_copy = True
        else:
            need_copy = False
        
        # 使用随机状态确保再次前向传播时生成相同的随机数
        with rand_state:
            if with_grad:
                outputs = model(**mini_batch_input)
            else:
                with torch.no_grad():
                    outputs = model(**mini_batch_input)
            
            embeds = outputs.last_hidden_state[:, 0]
        
        return embeds, rand_state if need_copy else None
    
    def encode_all_minibatches(self, model, input_dict, with_grad=True, copy_random_state=True, random_states=None):
        batch_size = len(input_dict["input_ids"])
        all_embeds = []
        all_random_states = []
        
        for i, start_idx in enumerate(range(0, batch_size, self.mini_batch_size)):
            end_idx = min(start_idx + self.mini_batch_size, batch_size)
            current_random_state = None if random_states is None else random_states[i]
            
            embeds, random_state = self.encode_minibatch(
                model=model,
                input_dict=input_dict,
                start_idx=start_idx,
                end_idx=end_idx,
                with_grad=with_grad,
                rand_state=current_random_state
            )
            
            if not with_grad:
                # 如果是第一阶段，需要保存梯度
                embeds = embeds.detach().clone().requires_grad_(True)
            
            all_embeds.append(embeds)
            if copy_random_state:
                all_random_states.append(random_state)
        
        return all_embeds, all_random_states
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 先把所有保存，后面应该有用reps = [[anchor的所有微批次], [positive的所有微批次], [negative的所有微批次]]
        reps = []
        self.random_states = []
        # import pdb
        # pdb.set_trace()
        anchor_batch_size = len(inputs["anchor_input"]["input_ids"])
        
        # 处理问题输入
        anchor_embeds, anchor_random_states = self.encode_all_minibatches(
            model=model,
            input_dict=inputs["anchor_input"],
            with_grad=False,
            copy_random_state=True
        )
        reps.append(anchor_embeds)
        self.random_states.append(anchor_random_states)
        
        # 处理正例输入
        positive_embeds, positive_random_states = self.encode_all_minibatches(
            model=model,
            input_dict=inputs["positive_input"],
            with_grad=False,
            copy_random_state=True
        )
        reps.append(positive_embeds)
        self.random_states.append(positive_random_states)
        
        # 处理负例输入
        negative_embeds, negative_random_states = self.encode_all_minibatches(
            model=model,
            input_dict=inputs["negative_input"],
            with_grad=False,
            copy_random_state=True
        )
        reps.append(negative_embeds)
        self.random_states.append(negative_random_states)
        
        if torch.is_grad_enabled():
            # 对应步骤2: 计算损失并缓存梯度
            loss = self.calculate_loss_and_cache_gradients(reps, anchor_batch_size)
            
            # 步骤3: 注册反向传播函数
            loss.register_hook(lambda grad_output: self._backward_hook(
                grad_output=grad_output,
                model=model,
                inputs=inputs
            ))
        else:
            # 评估模式下直接计算损失，不需要梯度
            loss = self.calculate_loss_without_gradients(reps, anchor_batch_size)
        
        return loss
    
    def calculate_loss_and_cache_gradients(self, reps, batch_size):
        
        anchor_embeds = torch.cat(reps[0])  # (batch_size, hidden_dim)
        
        # 合并所有正例和负例
        candidate_embeds = torch.cat([torch.cat(reps[1]),torch.cat(reps[2])], dim=0)     
        labels = torch.arange(batch_size, device=anchor_embeds.device)
        
        losses = []
        for b in range(0, batch_size, self.mini_batch_size):
            e = min(b + self.mini_batch_size, batch_size)
            
            scores = F.cosine_similarity(anchor_embeds[b:e].unsqueeze(1),candidate_embeds.unsqueeze(0),dim=2) * self.grad_scale
            
            loss_fn = nn.CrossEntropyLoss()
            mini_batch_loss = loss_fn(scores, labels[b:e]) * len(scores) / batch_size
            
            mini_batch_loss.backward()
            losses.append(mini_batch_loss.detach())
        
        # 这里需要保持计算图
        final_loss = sum(losses).requires_grad_()
        
        # 缓存anchor，positive，negative的梯度
        self.cache = [
            [emb.grad.clone() for emb in reps[0]],  
            [emb.grad.clone() for emb in reps[1]],  
            [emb.grad.clone() for emb in reps[2]]  
        ]
        
        return final_loss
    
    def calculate_loss_without_gradients(self, reps, batch_size):
        
        anchor_embeds = torch.cat(reps[0])
        
        candidate_embeds = torch.cat([
            torch.cat(reps[1]),  
            torch.cat(reps[2])  
        ], dim=0)
        
        labels = torch.arange(batch_size, device=anchor_embeds.device)
        
        scores = F.cosine_similarity(
            anchor_embeds.unsqueeze(1),
            candidate_embeds.unsqueeze(0),
            dim=2
        ) * self.grad_scale
        
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(scores, labels)
        
        return loss
    
    def _backward_hook(self, grad_output, model, inputs):
        
        assert self.cache is not None #梯度缓存不应为空
        assert self.random_states is not None # 随机状态不应为空
        
        with torch.enable_grad():
            # 处理三种输入类型：anchor、positive 和 negative
            for i, (input_data, rand_states, gradients) in enumerate([
                (inputs["anchor_input"], self.random_states[0], self.cache[0]),
                (inputs["positive_input"], self.random_states[1], self.cache[1]),
                (inputs["negative_input"], self.random_states[2], self.cache[2])
            ]):
                for j, (rand_state, grad) in enumerate(zip(rand_states, gradients)):
                    start_idx = j * self.mini_batch_size
                    end_idx = min(start_idx + self.mini_batch_size, len(input_data["input_ids"]))
                    
                    # 进行第二次前向传播，保留计算图
                    embeds, _ = self.encode_minibatch(
                        model=model,
                        input_dict=input_data,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        with_grad=True,
                        rand_state=rand_state
                    )
                    
                    surrogate = torch.dot(embeds.flatten(), grad.flatten()) * grad_output
                    
                    # 最后一次反向传播不需要保留计算图
                    is_last = (i == 2 and j == len(rand_states) - 1)
                    surrogate.backward(retain_graph=not is_last)
            
        return None

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
        # 我的数据是{"anchor":"字符串1","positive":"字符串2","negetive":"字符串3"} 需要用这个函数转换成{"anchor":"字符串1","positive":"字符串2","negetive":["字符串3"]}
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
        save_total_limit=3,
        # max_steps=900,
        num_train_epochs=3,
        warmup_ratio=0.05,
        learning_rate=3e-5,
        weight_decay=0.01,
        gradient_accumulation_steps=16,
        per_device_train_batch_size=32,
        dataloader_num_workers=16,
        fp16=True,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    # 初始化 Trainer
    trainer = BiEncoderTrainerWithGradCache(
    model=model,
    args=training_args,
    tokenizer=query_tokenizer,
    train_dataset=dataset,
    data_collator=lambda batch: collate_fn(batch, query_tokenizer, context_tokenizer, max_length),
    mini_batch_size=16  # 设置微批处理大小，可以根据GPU内存调整
)
    # ddd=trainer.get_train_dataloader()
    # for d in ddd:
    #     import pdb
    #     pdb.set_trace()
    trainer.train()


