import numpy as np  
from scipy.spatial.distance import cosine  
from collections import defaultdict
from joblib import Parallel, delayed
from .base_recommender import BaseRecommender
from .baseline import PopularityRecommender

class UserCF(BaseRecommender):  
    """基于用户的协同过滤推荐算法"""  
    
    def __init__(self, k=20, n_jobs=6):  
        """初始化用户协同过滤推荐器  
        
        Args:  
            k: 相似用户数量  
        """  
        super().__init__("UserCF")  
        self.k = k  
        self.n_jobs = n_jobs
        self.user_item_matrix = None  
        self.similarity_matrix = None  
        self.user_mappings = None  
        self.item_mappings = None
        self.coldstart = PopularityRecommender()
        
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
        
        # 创建用户-物品矩阵  
        n_users = len(self.user_mappings['idx_to_id'])  
        n_items = len(self.item_mappings['idx_to_id'])  
        self.user_item_matrix = np.zeros((n_users, n_items))  
        
        for _, row in train_data.iterrows():  
            # 使用映射获取正确的索引  
            user_id = row['userId']  
            item_id = row['itemId']  
        
            # 确保使用整数索引  
            user_idx = self.user_mappings['id_to_idx'][user_id]  
            item_idx = self.item_mappings['id_to_idx'][item_id]  
        
            self.user_item_matrix[user_idx, item_idx] = row['rating']   
        
        # 计算用户相似度  
        self._compute_similarity()  
        
        self.coldstart.fit(train_data)

        
    def _compute_similarities_for_user(self, user_idx, start_idx=0):
        """计算一个用户与其他用户的相似度"""
        n_users = self.user_item_matrix.shape[0]
        similarities = np.zeros(n_users)
        
        # 只计算与索引大于start_idx的用户的相似度(避免重复计算)
        for other_idx in range(start_idx, n_users):
            if other_idx == user_idx:
                continue
                
            if np.count_nonzero(self.user_item_matrix[user_idx]) > 0 and np.count_nonzero(self.user_item_matrix[other_idx]) > 0:
                sim = 1 - cosine(self.user_item_matrix[user_idx], self.user_item_matrix[other_idx])
                
                # 处理NaN值
                if np.isnan(sim):
                    sim = 0
                
                similarities[other_idx] = sim
        
        return user_idx, similarities
        
#     def _compute_similarity(self):  
#         """计算用户之间的相似度"""  
#         n_users = self.user_item_matrix.shape[0]  
#         self.similarity_matrix = np.zeros((n_users, n_users))  
        
#         for i in range(n_users):  
#             for j in range(i+1, n_users):  
#                 # 使用余弦相似度  
#                 if np.count_nonzero(self.user_item_matrix[i]) > 0 and np.count_nonzero(self.user_item_matrix[j]) > 0:  
#                     sim = 1 - cosine(self.user_item_matrix[i], self.user_item_matrix[j])  
                    
#                     # 处理NaN值  
#                     if np.isnan(sim):  
#                         sim = 0  
                    
#                     self.similarity_matrix[i, j] = sim  
#                     self.similarity_matrix[j, i] = sim  

    
    def _compute_similarity(self):
        """使用joblib并行计算用户相似度"""
        n_users = self.user_item_matrix.shape[0]
        self.similarity_matrix = np.zeros((n_users, n_users))
        
        # 并行计算每个用户与其他用户的相似度
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._compute_similarities_for_user)(i, i+1) for i in range(n_users)
        )
        
        # 填充相似度矩阵
        for user_idx, similarities in results:
            for other_idx in range(user_idx+1, n_users):
                sim = similarities[other_idx]
                self.similarity_matrix[user_idx, other_idx] = sim
                self.similarity_matrix[other_idx, user_idx] = sim  # 对称矩阵
    
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
            return self.coldstart.recommend(user_id, n_recommendations) 
            
        user_idx = self.user_mappings['id_to_idx'][user_id]  
        
        # 获取最相似的k个用户  
        similar_users = np.argsort(self.similarity_matrix[user_idx])[::-1][:self.k]  
        
        # 为用户计算物品评分预测  
        item_scores = defaultdict(float)  
        for similar_user in similar_users:  
            similarity = self.similarity_matrix[user_idx, similar_user]  
            
            if similarity <= 0:  
                continue  
                
            for item_idx in range(self.user_item_matrix.shape[1]):  
                # 如果相似用户评价过这个物品  
                if self.user_item_matrix[similar_user, item_idx] > 0:  
                    # 如果需要排除已知物品  
                    if exclude_known and self.user_item_matrix[user_idx, item_idx] > 0:  
                        continue  
                    
                    # 加权评分  
                    item_scores[item_idx] += similarity * self.user_item_matrix[similar_user, item_idx]  
        
        # 排序并返回前N个推荐  
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)  
        top_items = sorted_items[:n_recommendations]  
        
        # 转换回原始物品ID  
        recommendations = [self.item_mappings['idx_to_id'][item_idx] for item_idx, _ in top_items]  
        
        return recommendations  