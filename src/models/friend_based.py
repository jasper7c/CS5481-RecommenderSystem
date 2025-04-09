import numpy as np
from collections import defaultdict
from .base_recommender import BaseRecommender
from .baseline import PopularityRecommender

class FriendBasedCF(BaseRecommender):
    """基于用户朋友的协同过滤推荐算法"""
    
    def __init__(self, friend_weight=0.5):
        """初始化基于朋友的协同过滤推荐器
        
        Args:
            friend_weight: 朋友评分的权重系数
        """
        super().__init__("FriendBasedCF")
        self.friend_weight = friend_weight
        self.user_item_matrix = None
        self.friend_network = defaultdict(list)  # 用户朋友关系网络
        self.user_mappings = None
        self.item_mappings = None
        self.coldstart = PopularityRecommender()
        
    def fit(self, rating_data, friend_data):
        """训练模型
        
        Args:
            rating_data: 评分数据，包含userId, itemId, rating字段
            friend_data: 朋友关系数据，包含userID, friendID字段
        """
        # 保存用户和物品的映射
        self.user_mappings = {
            'idx_to_id': dict(enumerate(rating_data['userId'].unique())),
            'id_to_idx': {id: idx for idx, id in enumerate(rating_data['userId'].unique())}
        }
        
        self.item_mappings = {
            'idx_to_id': dict(enumerate(rating_data['itemId'].unique())),
            'id_to_idx': {id: idx for idx, id in enumerate(rating_data['itemId'].unique())}
        }
        
        # 创建用户-物品矩阵
        n_users = len(self.user_mappings['idx_to_id'])
        n_items = len(self.item_mappings['idx_to_id'])
        self.user_item_matrix = np.zeros((n_users, n_items))
        
        self.coldstart.fit(rating_data)
        
        for _, row in rating_data.iterrows():
            # 使用映射获取正确的索引
            user_id = row['userId']
            item_id = row['itemId']
            
            # 确保用户和物品ID在映射中
            if user_id in self.user_mappings['id_to_idx'] and item_id in self.item_mappings['id_to_idx']:
                user_idx = self.user_mappings['id_to_idx'][user_id]
                item_idx = self.item_mappings['id_to_idx'][item_id]
                self.user_item_matrix[user_idx, item_idx] = row['rating']
        
        # 构建朋友关系网络
        for _, row in friend_data.iterrows():
            user_id = row['userId']
            friend_id = row['friendId']
            
            # 确保用户和朋友ID在映射中
            if user_id in self.user_mappings['id_to_idx'] and friend_id in self.user_mappings['id_to_idx']:
                user_idx = self.user_mappings['id_to_idx'][user_id]
                friend_idx = self.user_mappings['id_to_idx'][friend_id]
                self.friend_network[user_idx].append(friend_idx)
    
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
        
        # 获取用户的朋友
        friends = self.friend_network.get(user_idx, [])
        
        if not friends:
            return []  # 如果用户没有朋友，返回空列表
        
        # 用户已评分的物品
        user_rated_items = set()
        if exclude_known:
            for item_idx in range(self.user_item_matrix.shape[1]):
                if self.user_item_matrix[user_idx, item_idx] > 0:
                    user_rated_items.add(item_idx)
        
        # 计算朋友对物品的平均评分
        item_scores = defaultdict(float)
        item_count = defaultdict(int)
        
        for friend_idx in friends:
            for item_idx in range(self.user_item_matrix.shape[1]):
                # 如果朋友评价过这个物品
                if self.user_item_matrix[friend_idx, item_idx] > 0:
                    # 如果需要排除用户已知物品
                    if item_idx in user_rated_items:
                        continue
                    
                    # 累加朋友的评分
                    item_scores[item_idx] += self.user_item_matrix[friend_idx, item_idx]
                    item_count[item_idx] += 1
        
        # 计算加权平均评分
        weighted_scores = {}
        for item_idx, total_score in item_scores.items():
            if item_count[item_idx] > 0:
                # 朋友评分的加权平均
                avg_score = total_score / item_count[item_idx]
                
                # 结合用户自己的评分模式（如果有相关数据可以添加）
                weighted_scores[item_idx] = self.friend_weight * avg_score
                
        
        # 排序并返回前N个推荐
        sorted_items = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:n_recommendations]
        
        # 转换回原始物品ID
        recommendations = [self.item_mappings['idx_to_id'][item_idx] for item_idx, _ in top_items]
        
        return recommendations
    
    def recommend_with_explanation(self, user_id, n_recommendations=10, exclude_known=True):
        """为用户生成带解释的推荐
        
        Args:
            user_id: 用户ID
            n_recommendations: 推荐数量
            exclude_known: 是否排除已知物品
            
        Returns:
            包含推荐物品ID和解释的列表
        """
        if user_id not in self.user_mappings['id_to_idx']:
            return []
            
        user_idx = self.user_mappings['id_to_idx'][user_id]
        
        # 获取用户的朋友
        friends = self.friend_network.get(user_idx, [])
        
        if not friends:
            return []  # 如果用户没有朋友，返回空列表
        
        # 用户已评分的物品
        user_rated_items = set()
        if exclude_known:
            for item_idx in range(self.user_item_matrix.shape[1]):
                if self.user_item_matrix[user_idx, item_idx] > 0:
                    user_rated_items.add(item_idx)
        
        # 计算朋友对物品的评分情况
        item_scores = defaultdict(float)
        item_count = defaultdict(int)
        item_friends = defaultdict(list)  # 记录评分的朋友
        
        for friend_idx in friends:
            for item_idx in range(self.user_item_matrix.shape[1]):
                if self.user_item_matrix[friend_idx, item_idx] > 0:
                    if item_idx in user_rated_items:
                        continue
                    
                    item_scores[item_idx] += self.user_item_matrix[friend_idx, item_idx]
                    item_count[item_idx] += 1
                    
                    # 记录哪些朋友评价了该物品
                    friend_id = self.user_mappings['idx_to_id'][friend_idx]
                    item_friends[item_idx].append((friend_id, self.user_item_matrix[friend_idx, item_idx]))
        
        # 计算加权平均评分
        weighted_scores = {}
        for item_idx, total_score in item_scores.items():
            if item_count[item_idx] > 0:
                avg_score = total_score / item_count[item_idx]
                weighted_scores[item_idx] = self.friend_weight * avg_score
        
        # 排序并返回前N个推荐
        sorted_items = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:n_recommendations]
        
        # 构建带解释的推荐结果
        recommendations_with_explanation = []
        for item_idx, score in top_items:
            item_id = self.item_mappings['idx_to_id'][item_idx]
            friend_ratings = item_friends[item_idx]
            
            # 构建解释
            explanation = {
                "item_id": item_id,
                "predicted_score": score,
                "friend_ratings": friend_ratings
            }
            
            recommendations_with_explanation.append(explanation)
        
        return recommendations_with_explanation