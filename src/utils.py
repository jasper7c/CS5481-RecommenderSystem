import pandas as pd
import time

def run_recommendation_model(model, train_df, test_df, movies_df, ratings_df, evaluator, 
                            sample_user_id=None, sample_movie_id=None):
    """
    运行推荐模型的通用函数，包括训练、评估和推荐
    
    参数:
    - model: 推荐模型实例
    - train_df: 训练数据集
    - test_df: 测试数据集
    - movies_df: 电影数据集
    - ratings_df: 评分数据集
    - evaluator: 评估器实例
    - sample_user_id: 样本用户ID (可选)
    - sample_movie_id: 样本电影ID (可选，用于内容推荐)
    
    返回:
    - results: 评估结果
    """
    # 确定模型名称
    model_name = model.name
    
    # 1. 训练模型
    print(f"\n=== Training {model_name} Model ===")
    print(f"Training {model_name} model...")
    start_time = time.time()
    
    # 根据模型类型使用不同的训练方法
    if hasattr(model, 'fit') and model.__class__.__name__ == 'ContentBased':
        model.fit(train_df, movies_df)
    else:
        model.fit(train_df)
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # 2. 评估模型
    print(f"\n=== Evaluating {model_name} Model ===")
    start_time = time.time()
    results = evaluator.evaluate(model, k_values=[5, 10, 20])
    eval_time = time.time() - start_time
    
    # 打印评估结果
    print(f"\n{model_name} Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    
    # 3. 为样本用户生成推荐
    print(f"\n=== {model_name} Sample Recommendations ===")
    
    # 如果未提供样本用户，使用训练集中的第一个用户
    if sample_user_id is None:
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
    recommendations = model.recommend(sample_user_id, n_recommendations=10)
    
    # 获取推荐电影的详细信息
    recommended_movies = movies_df[movies_df['movieId'].isin(recommendations)]
    print(f"\n{model_name} Top 10 recommendations for user {sample_user_id}:")
    for _, movie in recommended_movies.iterrows():
        print(f"- {movie['title']} ({movie['genres']})")
    
    # 4. 如果是基于内容的推荐，添加相似电影推荐
    if hasattr(model, 'get_similar_movies'):
        # 如果未提供样本电影，使用第一个电影
        if sample_movie_id is None:
            sample_movie_id = movies_df['movieId'].iloc[0]
        
        sample_movie_title = movies_df.loc[movies_df['movieId'] == sample_movie_id, 'title'].iloc[0]
        print(f"\nMovies similar to '{sample_movie_title}':")
        
        similar_movies = model.get_similar_movies(sample_movie_id, n=5)
        similar_movies_df = movies_df[movies_df['movieId'].isin(similar_movies)]
        
        for _, movie in similar_movies_df.iterrows():
            print(f"- {movie['title']} ({movie['genres']})")
    
    return results