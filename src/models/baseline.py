import numpy as np
from collections import defaultdict
from .base_recommender import BaseRecommender


class RandomRecommender(BaseRecommender):
    """随机推荐算法"""

    def __init__(self, name="RandomRecommender"):
        super().__init__(name)
        self.item_set = None
        self.user_interactions = None

    def fit(self, train_data):
        """训练模型

        Args:
            train_data: 训练数据，包含user_id, item_id字段
        """
        # 获取所有物品的集合
        self.item_set = set(train_data['itemId'].unique())

        # 记录每个用户已交互的物品
        self.user_interactions = defaultdict(set)
        for _, row in train_data.iterrows():
            user_id = row['userId']
            item_id = row['itemId']
            self.user_interactions[user_id].add(item_id)

    def recommend(self, user_id, n_recommendations=10, exclude_known=True):
        """为用户生成随机推荐

        Args:
            user_id: 用户ID
            n_recommendations: 推荐数量
            exclude_known: 是否排除已知项目

        Returns:
            推荐的物品ID列表
        """
        # 获取所有物品的列表
        items = list(self.item_set)

        # 排除已知物品
        if exclude_known and user_id in self.user_interactions:
            known_items = self.user_interactions[user_id]
            items = [item for item in items if item not in known_items]

        # 随机选择推荐物品
        np.random.shuffle(items)
        return items[:n_recommendations]


class PopularityRecommender(BaseRecommender):
    """基于流行度的推荐算法"""

    def __init__(self, name="PopularityRecommender"):
        super().__init__(name)
        self.popular_items = None
        self.user_interactions = None

    def fit(self, train_data):
        """训练模型

        Args:
            train_data: 训练数据，包含user_id, itemId, rating字段
        """
        # 计算每个物品的流行度（评分总和或出现次数）
        item_popularity = train_data.groupby('itemId')['rating'].sum().reset_index()
        item_popularity = item_popularity.sort_values('rating', ascending=False)
        self.popular_items = item_popularity['itemId'].tolist()

        # 记录每个用户已交互的物品
        self.user_interactions = defaultdict(set)
        for _, row in train_data.iterrows():
            user_id = row['userId']
            item_id = row['itemId']
            self.user_interactions[user_id].add(item_id)

    def recommend(self, user_id, n_recommendations=10, exclude_known=True):
        """为用户生成基于流行度的推荐

        Args:
            user_id: 用户ID
            n_recommendations: 推荐数量
            exclude_known: 是否排除已知项目

        Returns:
            推荐的物品ID列表
        """
        # 获取流行物品列表
        recommendations = []
        for item in self.popular_items:
            # 排除已知物品
            if exclude_known and user_id in self.user_interactions and item in self.user_interactions[user_id]:
                continue
            recommendations.append(item)
            if len(recommendations) >= n_recommendations:
                break
        return recommendations