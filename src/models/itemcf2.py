import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict
from joblib import Parallel, delayed
from .base_recommender import BaseRecommender

class ItemCF(BaseRecommender):
    """基于物品的协同过滤推荐算法"""
    
    def __init__(self, k=20, n_jobs=6):
        """初始化物品协同过滤推荐器
        
        Args:
            k: 相似物品数量
            n_jobs: 并行计算的CPU核心数
        """
        super().__init__("ItemCF")
        self.k = k
        self.n_jobs = n_jobs
        self.item_user_matrix = None  # 物品-用户矩阵
        self.similarity_matrix = None  # 物品相似度矩阵
        self.user_mappings = None  # 用户ID到索引的映射
        self.item_mappings = None  # 物品ID到索引的映射
    
    def fit(self, train_data):
        """训练模型
        
        Args:
            train_data: 训练数据，包含user_idx, item_idx, rating字段
        """
        # 保存用户和物品的映射
        self.user_mappings = {
            'idx_to_id': dict(enumerate(train_data['userId'].unique())),
            'id_to_idx': {id: idx for idx, id in enumerate(train_data['userId'].unique())}
        }
        
        self.item_mappings = {
            'idx_to_id': dict(enumerate(train_data['itemId'].unique())),
            'id_to_idx': {id: idx for idx, id in enumerate(train_data['itemId'].unique())}
        }
        
        # 创建物品-用户矩阵
        n_items = len(self.item_mappings['idx_to_id'])
        n_users = len(self.user_mappings['idx_to_id'])
        self.item_user_matrix = np.zeros((n_items, n_users))
        
        for _, row in train_data.iterrows():
            # 使用映射获取正确的索引
            user_id = row['userId']
            item_id = row['itemId']
            
            # 确保使用整数索引
            user_idx = self.user_mappings['id_to_idx'][user_id]
            item_idx = self.item_mappings['id_to_idx'][item_id]
            
            self.item_user_matrix[item_idx, user_idx] = row['rating']
        
        # 计算物品相似度
        self._compute_similarity()
    
    def _compute_similarities_for_item(self, item_idx, start_idx=0):
        """计算一个物品与其他物品的相似度"""
        n_items = self.item_user_matrix.shape[0]
        similarities = np.zeros(n_items)
        
        # 只计算与索引大于start_idx的物品的相似度(避免重复计算)
        for other_idx in range(start_idx, n_items):
            if other_idx == item_idx:
                continue
                
            if np.count_nonzero(self.item_user_matrix[item_idx]) > 0 and np.count_nonzero(self.item_user_matrix[other_idx]) > 0:
                sim = 1 - cosine(self.item_user_matrix[item_idx], self.item_user_matrix[other_idx])
                
                # 处理NaN值
                if np.isnan(sim):
                    sim = 0
                
                similarities[other_idx] = sim
        
        return item_idx, similarities
    
    def _compute_similarity(self):
        """使用joblib并行计算物品相似度"""
        n_items = self.item_user_matrix.shape[0]
        self.similarity_matrix = np.zeros((n_items, n_items))
        
        # 并行计算每个物品与其他物品的相似度
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._compute_similarities_for_item)(i, i+1) for i in range(n_items)
        )
        
        # 填充相似度矩阵
        for item_idx, similarities in results:
            for other_idx in range(item_idx+1, n_items):
                sim = similarities[other_idx]
                self.similarity_matrix[item_idx, other_idx] = sim
                self.similarity_matrix[other_idx, item_idx] = sim  # 对称矩阵
    
    def recommend(self, user_id, n_recommendations=10, exclude_known=True):
        """为用户生成推荐
        
        Args:
            user_id: 用户ID
            n_recommendations: 推荐数量
            exclude_known: 是否排除已知物品
            
        Returns:
            推荐的物品ID列表
        """
        if user_id not in self.user_mappings['id_to_idx']:
            return []
            
        user_idx = self.user_mappings['id_to_idx'][user_id]
        
        # 获取用户评分过的物品
        rated_items = np.where(self.item_user_matrix[:, user_idx] > 0)[0]
        
        # 如果用户没有评分过任何物品，返回空列表
        if len(rated_items) == 0:
            return []
        
        # 为用户计算物品评分预测
        item_scores = defaultdict(float)
        for item_idx in rated_items:
            rating = self.item_user_matrix[item_idx, user_idx]
            
            # 获取与该物品最相似的k个物品
            similar_items = np.argsort(self.similarity_matrix[item_idx])[::-1][:self.k]
            
            for similar_item in similar_items:
                similarity = self.similarity_matrix[item_idx, similar_item]
                
                if similarity <= 0:
                    continue
                
                # 如果需要排除已知物品
                if exclude_known and self.item_user_matrix[similar_item, user_idx] > 0:
                    continue
                
                # 加权评分
                item_scores[similar_item] += similarity * rating
        
        # 排序并返回前N个推荐
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:n_recommendations]
        
        # 转换回原始物品ID
        recommendations = [self.item_mappings['idx_to_id'][item_idx] for item_idx, _ in top_items]
        
        return recommendations