import numpy as np
from scipy.stats import pearsonr
from collections import defaultdict
from .base_recommender import BaseRecommender


class ItemCF(BaseRecommender):
    """基于物品的协同过滤推荐算法"""

    def __init__(self, k=20):
        """初始化物品协同过滤推荐器

        Args:
            k: 相似物品数量
        """
        super().__init__("ItemCF")
        self.k = k
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_mappings = None
        self.item_mappings = None

    def fit(self, train_data):
        """训练模型

        Args:
            train_data: 训练数据，包含userId, movieId, rating字段
        """
        # 保存用户和物品的映射
        self.user_mappings = {
            'idx_to_id': dict(enumerate(train_data['userId'].unique())),
            'id_to_idx': {id: idx for idx, id in enumerate(train_data['userId'].unique())}
        }

        self.item_mappings = {
            'idx_to_id': dict(enumerate(train_data['movieId'].unique())),
            'id_to_idx': {id: idx for idx, id in enumerate(train_data['movieId'].unique())}
        }

        # 创建用户-物品矩阵
        n_users = len(self.user_mappings['idx_to_id'])
        n_items = len(self.item_mappings['idx_to_id'])
        self.user_item_matrix = np.zeros((n_users, n_items))

        for _, row in train_data.iterrows():
            user_id = row['userId']
            item_id = row['movieId']
            user_idx = self.user_mappings['id_to_idx'][user_id]
            item_idx = self.item_mappings['id_to_idx'][item_id]
            self.user_item_matrix[user_idx, item_idx] = row['rating']

        # 计算物品相似度
        self._compute_similarity()

    def _compute_similarity(self):
        """计算物品之间的相似度"""
        n_items = self.user_item_matrix.shape[1]
        self.similarity_matrix = np.zeros((n_items, n_items))

        for i in range(n_items):
            for j in range(i + 1, n_items):
                # 获取物品i和物品j的评分向量
                item_i_ratings = self.user_item_matrix[:, i]
                item_j_ratings = self.user_item_matrix[:, j]

                # 获取共同评分的用户索引
                common_users = (item_i_ratings > 0) & (item_j_ratings > 0)
                if np.sum(common_users) < 2:
                    continue

                # 提取共同评分
                ratings_i = item_i_ratings[common_users]
                ratings_j = item_j_ratings[common_users]

                # 检查是否为常量
                if np.std(ratings_i) == 0 or np.std(ratings_j) == 0:
                    self.similarity_matrix[i, j] = 0
                    self.similarity_matrix[j, i] = 0
                    continue

                # 计算皮尔逊相关系数
                corr, _ = pearsonr(ratings_i, ratings_j)

                # 处理NaN值
                if np.isnan(corr):
                    corr = 0

                self.similarity_matrix[i, j] = corr
                self.similarity_matrix[j, i] = corr

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
        user_ratings = self.user_item_matrix[user_idx]

        # 初始化物品评分预测
        item_scores = defaultdict(float)

        # 获取用户已评分的物品
        rated_items = np.where(user_ratings > 0)[0]

        for item_idx in rated_items:
            # 获取相似物品
            similar_items = np.argsort(self.similarity_matrix[item_idx])[::-1][:self.k]

            for sim_item_idx in similar_items:
                similarity = self.similarity_matrix[item_idx, sim_item_idx]
                if similarity <= 0:
                    continue

                # 如果需要排除已知物品
                if exclude_known and self.user_item_matrix[user_idx, sim_item_idx] > 0:
                    continue

                # 加权评分
                item_scores[sim_item_idx] += similarity * user_ratings[item_idx]

        # 排序并返回前N个推荐
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:n_recommendations]

        # 转换回原始物品ID
        recommendations = [self.item_mappings['idx_to_id'][item_idx] for item_idx, _ in top_items]

        return recommendations