import numpy as np  
import pandas as pd  
from collections import defaultdict  

class Evaluator:  
    """评估推荐系统性能的类"""  
    
    def __init__(self, test_data):  
        """初始化评估器  
        
        Args:  
            test_data: 测试数据集  
        """  
        self.test_data = test_data  
        self._prepare_test_data()  
        
    def _prepare_test_data(self):  
        """准备测试数据，按用户分组"""  
        self.user_items = defaultdict(list)  
        
        for _, row in self.test_data.iterrows():  
            user_id = row['userId']  
            item_id = row['itemId']  
            self.user_items[user_id].append(item_id)  
    
    def hr_at_k(self, recommendations, k=10):  
        """计算HR@K (Hit Ratio at K)  
        
        Args:  
            recommendations: {user_id: [item_id1, item_id2, ...]} 格式的推荐结果  
            k: 推荐列表长度  
            
        Returns:  
            HR@K 值  
        """  
        hits = 0  
        total_users = 0  
        
        for user_id, ground_truth in self.user_items.items():  
            if user_id not in recommendations:  
                continue  
                
            recs = recommendations[user_id][:k]  
            # 如果推荐列表中有一个物品在真实列表中，就算命中  
            hit = any(item in ground_truth for item in recs)  
            hits += hit  
            total_users += 1  
            
        return hits / total_users if total_users > 0 else 0  
    
    def ndcg_at_k(self, recommendations, k=10):  
        """计算NDCG@K (Normalized Discounted Cumulative Gain at K)  
        
        Args:  
            recommendations: {user_id: [item_id1, item_id2, ...]} 格式的推荐结果  
            k: 推荐列表长度  
            
        Returns:  
            NDCG@K 值  
        """  
        ndcg_sum = 0  
        total_users = 0  
        
        for user_id, ground_truth in self.user_items.items():  
            if user_id not in recommendations:  
                continue  
                
            recs = recommendations[user_id][:k]  
            
            # 计算DCG  
            dcg = 0  
            for i, item in enumerate(recs):  
                if item in ground_truth:  
                    # 推荐物品的位置从0开始，所以i+1  
                    dcg += 1 / np.log2(i + 2)  
            
            # 计算IDCG (理想情况下的DCG)  
            # 在理想情况下，所有相关物品都排在前面  
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))  
            
            if idcg > 0:  
                ndcg_sum += dcg / idcg  
                
            total_users += 1  
            
        return ndcg_sum / total_users if total_users > 0 else 0

    def mrr_at_k(self, recommendations, k=10):
        """计算MRR@K (Mean Reciprocal Rank at K)

        Args:
            recommendations: {user_id: [item_id1, item_id2, ...]} 格式的推荐结果
            k: 推荐列表长度

        Returns:
            MRR@K 值
        """
        mrr_sum = 0
        total_users = 0

        for user_id, ground_truth in self.user_items.items():
            if user_id not in recommendations:
                continue

            recs = recommendations[user_id][:k]

            # 找到第一个相关物品的排名
            rank = 0
            for i, item in enumerate(recs):
                if item in ground_truth:
                    rank = i + 1  # 排名从1开始
                    break

            # 计算 Reciprocal Rank
            if rank > 0:
                mrr_sum += 1 / rank
            else:
                mrr_sum += 0

            total_users += 1

        return mrr_sum / total_users if total_users > 0 else 0

    def recall_at_k(self, recommendations, k=10):
        """计算Recall@K (召回率@K)

        Args:
            recommendations: {user_id: [item_id1, item_id2, ...]} 格式的推荐结果
            k: 推荐列表长度

        Returns:
            Recall@K 值
        """
        recall_sum = 0
        total_users = 0

        for user_id, ground_truth in self.user_items.items():
            if user_id not in recommendations:
                continue

            # 获取推荐列表的前K个物品
            recs = recommendations[user_id][:k]

            # 将推荐列表和真实列表转换为集合，方便计算交集
            recs_set = set(recs)
            ground_truth_set = set(ground_truth)

            # 计算推荐列表中与真实相关物品的重叠数量
            hit_count = len(recs_set & ground_truth_set)

            # 计算Recall
            if len(ground_truth_set) > 0:
                recall = hit_count / len(ground_truth_set)
                recall_sum += recall

            total_users += 1

        return recall_sum / total_users if total_users > 0 else 0
    
    def evaluate(self, recommender, k_values=[5, 10, 20]):  
        """评估推荐系统在多个K值下的性能  
        
        Args:  
            recommender: 推荐器对象  
            k_values: 要评估的K值列表  
            
        Returns:  
            评估结果字典  
        """  
        # 为测试集中的所有用户生成推荐  
        recommendations = {}  
        max_k = max(k_values)  
        
        for user_id in self.user_items.keys():  
            # 获取推荐结果  
            user_recs = recommender.recommend(user_id, n_recommendations=max_k)  
            recommendations[user_id] = user_recs  
        
        # 计算不同K值下的评估指标  
        results = {}  
        for k in k_values:  
            hr = self.hr_at_k(recommendations, k)  
            ndcg = self.ndcg_at_k(recommendations, k)
            mrr = self.mrr_at_k(recommendations, k)
            recall = self.recall_at_k(recommendations, k)
            
            results[f'HR@{k}'] = hr  
            results[f'NDCG@{k}'] = ndcg
            results[f'MRR@{k}'] = mrr
            results[f'Recall@{k}'] = recall
        
        return results  