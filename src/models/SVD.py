import numpy as np
from collections import defaultdict
from .base_recommender import BaseRecommender

class SVDRecommender(BaseRecommender):
    """基于SVD矩阵分解的推荐算法"""
    
    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02):
        """初始化SVD推荐器
        
        Args:
            n_factors: 潜在因子维度
            n_epochs: 训练轮数
            lr: 学习率
            reg: 正则化系数
        """
        super().__init__("SVD")
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.user_factors = None
        self.item_factors = None
        self.user_mappings = None
        self.item_mappings = None
        self.user_items = None  # 记录用户已交互物品

    def fit(self, train_data):
        """训练模型
        
        Args:
            train_data: 包含userId, movieId, rating的DataFrame
        """
        # 创建用户和物品的映射
        self.user_mappings = {
            'idx_to_id': dict(enumerate(train_data['userId'].unique())),
            'id_to_idx': {id: idx for idx, id in enumerate(train_data['userId'].unique())}
        }
        self.item_mappings = {
            'idx_to_id': dict(enumerate(train_data['movieId'].unique())),
            'id_to_idx': {id: idx for idx, id in enumerate(train_data['movieId'].unique())}
        }

        n_users = len(self.user_mappings['idx_to_id'])
        n_items = len(self.item_mappings['idx_to_id'])

        # 初始化潜在因子矩阵（正态分布）
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # 记录用户已评分的物品
        self.user_items = defaultdict(set)
        for _, row in train_data.iterrows():
            user_idx = self.user_mappings['id_to_idx'][row['userId']]
            item_idx = self.item_mappings['id_to_idx'][row['movieId']]
            self.user_items[user_idx].add(item_idx)

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_samples = 0
            
            for _, row in train_data.iterrows():
                # 获取数据（同上）
                user_idx = self.user_mappings['id_to_idx'][row['userId']]
                item_idx = self.item_mappings['id_to_idx'][row['movieId']]
                rating = row['rating']

                # 计算预测和误差
                pred = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
                error = rating - pred

                # 计算当前样本损失（包含正则化）
                sample_loss = error**2 
                sample_loss += self.reg * np.sum(self.user_factors[user_idx]**2)
                sample_loss += self.reg * np.sum(self.item_factors[item_idx]**2)
                epoch_loss += sample_loss
                n_samples += 1

                # 参数更新（同上）
                u_grad = error * self.item_factors[item_idx] - self.reg * self.user_factors[user_idx]
                i_grad = error * self.user_factors[user_idx] - self.reg * self.item_factors[item_idx]
                self.user_factors[user_idx] += self.lr * u_grad
                self.item_factors[item_idx] += self.lr * i_grad

            # 打印epoch日志
            avg_loss = epoch_loss / n_samples
            print(f"[Epoch {epoch+1:>2}/{self.n_epochs}] Loss: {avg_loss:.4f}")

    def recommend(self, user_id, n_recommendations=10, exclude_known=True):
        """生成推荐列表
        
        Args:
            user_id: 要推荐的用户ID
            n_recommendations: 推荐数量
            exclude_known: 是否排除用户已知道的物品
            
        Returns:
            推荐物品ID列表
        """
        if user_id not in self.user_mappings['id_to_idx']:
            return []
        
        user_idx = self.user_mappings['id_to_idx'][user_id]
        
        # 计算所有物品的预测评分
        scores = np.dot(self.user_factors[user_idx], self.item_factors.T)
        
        # 获取已评分的物品
        rated_items = self.user_items[user_idx] if exclude_known else set()
        
        # 生成推荐候选
        recommendations = []
        for item_idx in np.argsort(scores)[::-1]:
            if item_idx not in rated_items:
                item_id = self.item_mappings['idx_to_id'][item_idx]
                recommendations.append(item_id)
                if len(recommendations) >= n_recommendations:
                    break
        
        return recommendations