from data_processor import DataProcessor  
from models.user_cf import UserCF  
from models.content_based import ContentBased  
from evaluation import Evaluator  
import pandas as pd  
import time  
import os  
from utils import run_recommendation_model  

def main():  
    """主函数，运行完整的推荐系统流程"""  
    print("Starting MovieLens Recommender System...")  
    
    # 记录开始时间  
    start_time = time.time()  
    
    # 1. 加载和预处理数据  
    print("\n=== Data Processing ===")  
    data_processor = DataProcessor(data_path='data/raw/')  
    ratings_df, movies_df = data_processor.load_data()  
    
    # 预处理数据  
    processed_df = data_processor.preprocess(ratings_df)  
    
    # 划分训练集和测试集  
    train_df, test_df = data_processor.split_data(processed_df)  
    
    # 创建评估器
    evaluator = Evaluator(test_df)
    
    # 选择样本用户和样本电影
    sample_user_id = train_df['userId'].iloc[0]
    sample_movie_id = movies_df['movieId'].iloc[0]
    
    # 2. 训练并评估UserCF模型
    user_cf = UserCF(k=20)
    user_cf_results = run_recommendation_model(
        model=user_cf,
        train_df=train_df,
        test_df=test_df,
        movies_df=movies_df,
        ratings_df=ratings_df,
        evaluator=evaluator,
        sample_user_id=sample_user_id
    )
    
    # 3. 训练并评估ContentBased模型
    content_based = ContentBased()
    content_based_results = run_recommendation_model(
        model=content_based,
        train_df=train_df,
        test_df=test_df,
        movies_df=movies_df,
        ratings_df=ratings_df,
        evaluator=evaluator,
        sample_user_id=sample_user_id,
        sample_movie_id=sample_movie_id
    )
    
    # 输出总运行时间  
    elapsed_time = time.time() - start_time  
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds") 

if __name__ == "__main__":  
    main()