from data_processor import DataProcessor  
from models.user_cf import UserCF
from models.item_cf import ItemCF
from models.baseline import RandomRecommender
from models.baseline import PopularityRecommender
from models.SVD import SVDRecommender
from models.NGCN import NGCNRecommender
from models.LGCN import LightGCN
# from models.LightGCN import LightGCN
from evaluation import Evaluator  
import pandas as pd  
import time  
import os  
from models.content_based import ContentBased  

def main():  
    """主函数，运行完整的推荐系统流程"""  
    print("Starting MovieLens Recommender System...")  
    
    # 记录开始时间  
    start_time = time.time()  
    
    # 1. 加载和预处理数据  
    print("\n=== Data Processing ===")  
    # data_processor = DataProcessor(data_path='data/ml-1m/')
    data_processor = DataProcessor(data_path='../data/Yelp JSON/yelp_dataset/')
    # ratings_df, movies_df = data_processor.load_data()
    # ratings_df, movies_df, users_df = data_processor.load_dat_data()
    ratings_df, movies_df, users_df = data_processor.load_yelp_csv_data()
    # 预处理数据  
    processed_df = data_processor.preprocess(ratings_df)  
    
    # 划分训练集和测试集  
    train_df, test_df = data_processor.split_data(processed_df)  
    
    # 2. 训练推荐模型  
    print("\n=== Model Training ===")  
    # user_cf = UserCF(k=20, use_gpu=True)
    # user_cf = ItemCF(k=20)
    # user_cf = SVDRecommender(n_factors=100, n_epochs=50)
    user_cf = LightGCN(embed_dim=64, n_layers=3, epochs=20, lr=0.01)
    # user_cf = NGCNRecommender(epochs=30)
    # user_cf = RandomRecommender()
    print(f"Training {user_cf.name} model...")  
    user_cf.fit(train_df)  
    
    # 3. 评估模型  
    print("\n=== Model Evaluation ===")  
    evaluator = Evaluator(test_df)  
    results = evaluator.evaluate(user_cf, k_values=[5, 10, 20])  
    
    # 打印评估结果  
    print("\nEvaluation Results:")  
    for metric, value in results.items():  
        print(f"{metric}: {value:.4f}")  
    
    # 4. 为示例用户生成推荐  
    print("\n=== Sample Recommendations ===")  
    # 选择一个样本用户  
    sample_user_id = train_df['userId'].iloc[0]  
    print(f"Generating recommendations for user {sample_user_id}...")  
    
    # 获取用户已评分的电影  
    user_rated = ratings_df[ratings_df['userId'] == sample_user_id]  
    user_rated = pd.merge(user_rated, movies_df, on='movieId')  
    print(f"\nUser has rated {len(user_rated)} movies, top 5 highest rated:")  
    top_rated = user_rated.sort_values('rating', ascending=False).head(5)  
    for _, row in top_rated.iterrows():  
        print(f"- {row['title']} ({row['rating']})")  
    
    # 生成推荐  
    recommendations = user_cf.recommend(sample_user_id, n_recommendations=10)  
    
    # 获取推荐电影的详细信息  
    recommended_movies = movies_df[movies_df['movieId'].isin(recommendations)]  
    print("\nTop 10 recommendations:")  
    for _, movie in recommended_movies.iterrows():  
        print(f"- {movie['title']} ({movie['genres']})")     
    
    # 输出总运行时间  
    elapsed_time = time.time() - start_time  
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds") 

if __name__ == "__main__":  
    main()  