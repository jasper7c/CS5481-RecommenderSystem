import argparse
import time
from data_processor import DataProcessor
from models.user_cf import UserCF
from models.itemcf2 import ItemCF
from models.baseline import RandomRecommender
from models.baseline import PopularityRecommender
from models.content_based import ContentBased
from models.friend_based import FriendBasedCF
from models.SVD import SVDRecommender
from models.NGCN import NGCNRecommender
import pandas as pd
from models.LGCN import LightGCN
from evaluation import Evaluator
import sys
import os
import io
import random
import numpy as np
import torch

# 设置随机种子
seed = 42

# Python 标准库
random.seed(seed)

# NumPy
np.random.seed(seed)

# PyTorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # 如果使用 GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Tee:
    def __init__(self, filename, mode='w'):
        self.file = open(filename, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, text):
        self.file.write(text)
        self.stdout.write(text)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        self.file.close()
        
# 模型配置字典（模型类与所需参数）
MODEL_CONFIG = {
    'usercf': {
        'class': UserCF,
        'params': ['k']
    },
    'itemcf': {
        'class': ItemCF,
        'params': ['k']
    },
    'svd': {
        'class': SVDRecommender,
        'params': ['n_factors', 'n_epochs']
    },
    'lightgcn': {
        'class': LightGCN,
        'params': ['embed_dim', 'n_layers', 'epochs']
    },
    'ngcn': {
        'class': NGCNRecommender,
        'params': ['embed_dim', 'n_layers','epochs']
    },
    'random': {
        'class': RandomRecommender,
        'params': []
    },

    'popular': {
        'class': PopularityRecommender,
        'params': []
    },
    'content': {
        'class': ContentBased,
        'params': []
    },
    'friend': {
        'class': FriendBasedCF,
        'params': []
    }
}


def validate_args(args):
    """验证模型参数是否齐全"""
    model_name = args.model
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Invalid model: {model_name}")

    required_params = MODEL_CONFIG[model_name]['params']
    missing_params = [p for p in required_params if not getattr(args, p)]

    if missing_params:
        raise ValueError(f"Missing parameters for {model_name}: {missing_params}")


def main(args):
    """运行推荐系统实验主流程"""
    print("\n=== Starting Recommender System Experiment ===")
    start_time = time.time()

    # 1. 数据加载与预处理
    print("\n=== Data Loading & Preprocessing ===")
    data_processor = DataProcessor(data_path=args.data_path)

    # 根据数据集选择加载方法
    if args.dataset == 'ml-1m':
        ratings_df, movies_df, users_df = data_processor.load_dat_data()
    elif args.dataset == 'lastfm':
        ratings_df, user_friends_df, artists_df = data_processor.load_lastfm_data()

    processed_df = data_processor.preprocess(ratings_df)

    # 2. 数据划分
    print(f"\n=== Data Splitting ({args.split_method} method) ===")
    print(f"--- Loading 【{args.dataset}】 ---")
    if args.split_method == 'default':
        train_df, test_df = data_processor.split_data(
            processed_df,
            test_size=args.test_size,
            random_state=args.random_state
        )
    elif args.split_method == 'time':
        train_df, test_df = data_processor.split_data_by_time(
            processed_df,
            test_size=args.test_size
        )
    elif args.split_method == 'random':  # random
        train_df, test_df = data_processor.split_data_random(
            processed_df,
            test_size=args.test_size,
            random_state=args.random_state
        )

    # 3. 模型训练
    print("\n=== Model Training ===")
    model_info = MODEL_CONFIG[args.model]
    model_params = {p: getattr(args, p) for p in model_info['params']}
    recommender = model_info['class'](**model_params)

    print(f"Training {recommender.name} with params: {model_params}")
    
    if args.model == 'content':
        recommender.fit(train_df, movies_df)
    elif args.model == 'friend': 
        recommender.fit(train_df, user_friends_df)
    else:
        recommender.fit(train_df)

    # 4. 模型评估
    print("\n=== Model Evaluation ===")
    evaluator = Evaluator(test_df)
    results = evaluator.evaluate(recommender, k_values=args.k_values)

    # 打印结果
    print("\n=== Evaluation Results ===")
    for metric, value in results.items():  
        print(f"{metric}: {value:.4f}")  

    # 耗时统计
    print(f"\nTotal time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="推荐系统实验脚本")

    # 数据集参数
    parser.add_argument('--dataset', required=True,
                        choices=['ml-1m', 'lastfm'],
                        help="数据集名称")
    parser.add_argument('--data_path', required=True,
                        help="原始数据目录路径")

    # 数据划分参数
    parser.add_argument('--split_method', default='default',
                        choices=['default', 'time', 'random'],
                        help="数据划分方法")
    parser.add_argument('--test_size', type=float, default=0.2,
                        help="测试集比例")
    parser.add_argument('--random_state', type=int, default=42,
                        help="随机种子")

    # 模型参数
    parser.add_argument('--model', required=True,
                        choices=list(MODEL_CONFIG.keys()),
                        help="推荐算法类型")
    parser.add_argument('--k', type=int,
                        help="UserCF/ItemCF的邻居数量")
    parser.add_argument('--n_factors', type=int,
                        help="SVD的隐向量维度")
    parser.add_argument('--n_epochs', type=int,
                        help="SVD的训练轮数")
    parser.add_argument('--embed_dim', type=int,
                        help="LightGCN/NGCN的嵌入维度")
    parser.add_argument('--n_layers', type=int,
                        help="LightGCN/NGCN的层数")
    parser.add_argument('--epochs', type=int,
                        help="LightGCN/NGCN的训练轮数")

    # 评估参数
    parser.add_argument('--k_values', nargs='+', type=int,
                        default=[5, 10, 20],
                        help="评估指标K值列表")

    args = parser.parse_args()
    validate_args(args)
    # 生成带时间戳的日志文件
    log_filename = f"./log/experiment_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    with Tee(log_filename):
        main(args)