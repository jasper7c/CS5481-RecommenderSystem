import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  

class DataProcessor:  
    def __init__(self, data_path='data/raw/'):  
        """初始化数据处理器  

        Args:  
            data_path: 数据文件路径  
        """  
        self.data_path = data_path  
        
    def load_data(self):  
        """加载MovieLens数据集  

        Returns:  
            ratings_df: 评分数据  
            movies_df: 电影数据  
        """  
        # 加载评分数据  
        ratings_df = pd.read_csv(f"{self.data_path}ratings.csv")  
        # 加载电影数据  
        movies_df = pd.read_csv(f"{self.data_path}movies.csv")  
        
        print(f"Loaded ratings: {len(ratings_df)}, movies: {len(movies_df)}")  
        return ratings_df, movies_df  
    
    def preprocess(self, ratings_df):  
        """预处理评分数据  
        
        Args:  
            ratings_df: 原始评分数据  

        Returns:  
            处理后的评分数据  
        """  
        # 这里可以添加更多预处理步骤，如去除评分过少的用户和电影  
        # 目前只保留评分 >= 3.5 的作为正例  
        processed_df = ratings_df.copy()  
        processed_df['liked'] = (processed_df['rating'] >= 3.5).astype(int)  
        
        # 确保用户ID和电影ID是连续的整数  
        user_ids = processed_df['userId'].unique()  
        movie_ids = processed_df['movieId'].unique()  
        
        user_map = {old: new for new, old in enumerate(user_ids)}  
        movie_map = {old: new for new, old in enumerate(movie_ids)}  
        
        processed_df['user_idx'] = processed_df['userId'].map(user_map)  
        processed_df['movie_idx'] = processed_df['movieId'].map(movie_map)  
        
        return processed_df  
    
    def create_user_item_matrix(self, ratings_df):  
        """创建用户-物品评分矩阵  
        
        Args:  
            ratings_df: 预处理后的评分数据  
            
        Returns:  
            用户-物品评分矩阵  
        """  
        n_users = ratings_df['user_idx'].nunique()  
        n_items = ratings_df['movie_idx'].nunique()  
        
        # 创建矩阵  
        matrix = np.zeros((n_users, n_items))  
        
        for _, row in ratings_df.iterrows():  
            matrix[int(row['user_idx']), int(row['movie_idx'])] = row['rating']  
            
        return matrix  
    
    def split_data(self, ratings_df, test_size=0.2, random_state=42):  
        """划分训练集和测试集  
        
        Args:  
            ratings_df: 预处理后的评分数据  
            test_size: 测试集比例  
            random_state: 随机种子  
            
        Returns:  
            train_df: 训练集  
            test_df: 测试集  
        """  
        # 为每个用户划分训练集和测试集  
        train_data = []  
        test_data = []  
        
        for user_id in ratings_df['user_idx'].unique():  
            user_ratings = ratings_df[ratings_df['user_idx'] == user_id]  
            # 只考虑用户喜欢的电影  
            liked_items = user_ratings[user_ratings['liked'] == 1]  
            
            if len(liked_items) >= 2:  # 至少需要2个喜欢的电影才能划分  
                user_train, user_test = train_test_split(  
                    liked_items,   
                    test_size=test_size,   
                    random_state=random_state  
                )  
                train_data.append(user_train)  
                test_data.append(user_test)  
        
        train_df = pd.concat(train_data)  
        test_df = pd.concat(test_data)  
        
        print(f"Training set: {len(train_df)}, Test set: {len(test_df)}")  
        return train_df, test_df  