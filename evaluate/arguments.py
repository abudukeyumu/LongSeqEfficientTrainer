
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query-encoder-path', type=str, default='/mnt/abu/models/bge-large-en-v1.5')
    parser.add_argument('--context-encoder-path', type=str, default='/mnt/abu/models/bge-large-en-v1.5')
  
    parser.add_argument('--data-folder', type=str, default='', help='ChatRAG Bench的路径')
    parser.add_argument('--eval-dataset', type=str, default='', help='evaluation dataset (e.g., doc2dial)')

    parser.add_argument('--doc2dial-datapath', type=str, default='doc2dial/test.json')
    parser.add_argument('--doc2dial-docpath', type=str, default='doc2dial/documents.json')

    parser.add_argument('--quac-datapath', type=str, default='quac/test.json')
    parser.add_argument('--quac-docpath', type=str, default='quac/documents.json')
    
    parser.add_argument('--qrecc-datapath', type=str, default='qrecc/test.json')
    parser.add_argument('--qrecc-docpath', type=str, default='qrecc/documents.json')
    
    parser.add_argument('--topiocqa-datapath', type=str, default='/mnt/abu/evaluation/topiocqa/data/dev_with_idx.json')
    parser.add_argument('--topiocqa-docpath', type=str, default='/mnt/abu/evaluation/topiocqa/data/topiocqa_output.json')

    args = parser.parse_args()

    return args
