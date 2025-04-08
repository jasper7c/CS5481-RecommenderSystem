import numpy as np
import torch
from collections import defaultdict
from .base_recommender import BaseRecommender
from .baseline import PopularityRecommender

class SVDRecommender(BaseRecommender):
    """基于SVD矩阵分解的推荐算法 (GPU加速版)"""
    
    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02, device=None):
        """初始化SVD推荐器
        
        Args:
            n_factors: 潜在因子维度
            n_epochs: 训练轮数
            lr: 学习率
            reg: 正则化系数
            device: 运行设备，None表示自动选择，可以是'cpu'或'cuda'
        """
        super().__init__("SVD")
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.coldstart = PopularityRecommender()
        
        # 设置计算设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"使用设备: {self.device}")
        
        self.user_factors = None
        self.item_factors = None
        self.user_mappings = None
        self.item_mappings = None
        self.user_items = None  # 记录用户已交互物品

    def fit(self, train_data):
        """训练模型
        
        Args:
            train_data: 包含userId, itemId, rating的DataFrame
        """
        # 创建用户和物品的映射
        self.user_mappings = {
            'idx_to_id': dict(enumerate(train_data['userId'].unique())),
            'id_to_idx': {id: idx for idx, id in enumerate(train_data['userId'].unique())}
        }
        self.item_mappings = {
            'idx_to_id': dict(enumerate(train_data['itemId'].unique())),
            'id_to_idx': {id: idx for idx, id in enumerate(train_data['itemId'].unique())}
        }

        n_users = len(self.user_mappings['idx_to_id'])
        n_items = len(self.item_mappings['idx_to_id'])

        # 初始化潜在因子矩阵（正态分布）- 使用PyTorch
        self.user_factors = torch.normal(0, 0.1, (n_users, self.n_factors), device=self.device)
        self.item_factors = torch.normal(0, 0.1, (n_items, self.n_factors), device=self.device)
        
        self.coldstart.fit(train_data)

        # 记录用户已评分的物品
        self.user_items = defaultdict(set)
        for _, row in train_data.iterrows():
            user_idx = self.user_mappings['id_to_idx'][row['userId']]
            item_idx = self.item_mappings['id_to_idx'][row['itemId']]
            self.user_items[user_idx].add(item_idx)

        # 转换训练数据为张量以便批处理
        user_indices = torch.tensor([self.user_mappings['id_to_idx'][uid] for uid in train_data['userId']], 
                                    device=self.device)
        item_indices = torch.tensor([self.item_mappings['id_to_idx'][iid] for iid in train_data['itemId']], 
                                    device=self.device)
        ratings = torch.tensor(train_data['rating'].values, dtype=torch.float32, device=self.device)
        
        batch_size = 1024  # 设置批处理大小
        n_samples = len(ratings)
        
        for epoch in range(self.n_epochs):
            # 打乱数据顺序
            indices = torch.randperm(n_samples, device=self.device)
            user_indices = user_indices[indices]
            item_indices = item_indices[indices]
            ratings = ratings[indices]
            
            epoch_loss = 0.0
            
            # 批处理训练
            for i in range(0, n_samples, batch_size):
                batch_users = user_indices[i:i+batch_size]
                batch_items = item_indices[i:i+batch_size]
                batch_ratings = ratings[i:i+batch_size]
                
                # 获取当前批次的用户和物品因子
                u_factors = self.user_factors[batch_users]
                i_factors = self.item_factors[batch_items]
                
                # 计算预测和误差
                preds = torch.sum(u_factors * i_factors, dim=1)
                errors = batch_ratings - preds
                
                # 计算损失（包含正则化）
                batch_loss = torch.mean(errors ** 2)
                batch_loss += self.reg * (torch.mean(torch.sum(u_factors ** 2, dim=1)) + 
                                        torch.mean(torch.sum(i_factors ** 2, dim=1)))
                
                # 计算梯度更新
                u_grads = -errors.unsqueeze(1) * i_factors + self.reg * u_factors
                i_grads = -errors.unsqueeze(1) * u_factors + self.reg * i_factors
                
                # 更新因子
                self.user_factors[batch_users] -= self.lr * u_grads
                self.item_factors[batch_items] -= self.lr * i_grads
                
                epoch_loss += batch_loss.item() * len(batch_users) / n_samples

            # 打印epoch日志
            print(f"[Epoch {epoch+1:>2}/{self.n_epochs}] Loss: {epoch_loss:.4f}")
            
            

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
            return self.coldstart.recommend(user_id, n_recommendations)
        
        user_idx = self.user_mappings['id_to_idx'][user_id]
        
        # 计算所有物品的预测评分
        with torch.no_grad():  # 不计算梯度
            user_f = self.user_factors[user_idx].unsqueeze(0)  # [1, n_factors]
            scores = torch.mm(user_f, self.item_factors.t()).squeeze(0)  # [n_items]
            scores = scores.cpu().numpy()  # 转回CPU处理排序和过滤
        
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