from collections import defaultdict
from .base_recommender import BaseRecommender  # 或继承你原来的BaseRecommender
import pandas as pd

# python experiment.py --dataset yelp --data_path "../data/Yelp JSON/yelp_dataset/" --model citypopular --split_method default

class CityPopularityRecommender(BaseRecommender):
    def __init__(self, name="CityPopularityRecommender"):
        super().__init__(name)
        self.city_top_items = defaultdict(list)
        self.user_interactions = defaultdict(set)
        self.user_city = dict()

    def fit(self, train_df, business_df):
        # 构建用户-交互表
        for _, row in train_df.iterrows():
            self.user_interactions[row['userId']].add(row['itemId'])

        # 构建用户-城市映射（取其评分过的多数城市）
        merged = pd.merge(train_df, business_df[['itemId', 'city']], on='itemId', how='left')
        user_city_mode = merged.groupby('userId')['city'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
        self.user_city = user_city_mode.to_dict()

        # 构建城市-热门物品列表
        city_rating_sum = merged.groupby(['city', 'itemId'])['rating'].sum().reset_index()
        city_sorted = city_rating_sum.sort_values(['city', 'rating'], ascending=[True, False])
        for _, row in city_sorted.iterrows():
            self.city_top_items[row['city']].append(row['itemId'])

    def recommend(self, user_id, n_recommendations=10, exclude_known=True):
        city = self.user_city.get(user_id, 'Unknown')
        items = self.city_top_items.get(city, [])
        recommendations = []
        for item in items:
            if exclude_known and item in self.user_interactions[user_id]:
                continue
            recommendations.append(item)
            if len(recommendations) >= n_recommendations:
                break
        return recommendations
