import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
import csv


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
        ratings_df['itemId'] = ratings_df['movieId']
        # 加载电影数据  
        movies_df = pd.read_csv(f"{self.data_path}movies.csv")  
        
        print(f"Loaded ratings: {len(ratings_df)}, movies: {len(movies_df)}")  
        return ratings_df, movies_df

    def load_dat_data(self):
        """加载MovieLens-1M数据集

        Returns:
            ratings_df: 评分数据
            movies_df: 电影数据
            users_df: 用户数据
        """
        # 加载评分数据
        ratings_df = pd.read_csv(f"{self.data_path}ratings.dat", sep='::', engine='python', encoding='ISO-8859-1',
                                 names=['userId', 'movieId', 'rating', 'timestamp'])
        # 加载电影数据
        movies_df = pd.read_csv(f"{self.data_path}movies.dat", sep='::', engine='python', encoding='ISO-8859-1',
                                names=['movieId', 'title', 'genres'])
        # 加载用户数据
        users_df = pd.read_csv(f"{self.data_path}users.dat", sep='::', engine='python', encoding='ISO-8859-1',
                               names=['userId', 'gender', 'age', 'occupation', 'zip-code'])
        
        ratings_df['itemId'] = ratings_df['movieId']
        print(f"Loaded ratings: {len(ratings_df)}, movies: {len(movies_df)}, users: {len(users_df)}")
        return ratings_df, movies_df, users_df

    def load_lastfm_data(self):
        """加载Last.fm数据集的user_artists.dat, user_friends.dat和artists.dat文件

        Args:
            data_path (str): 数据集文件夹路径

        Returns:
            user_artists_df: 用户-艺术家关系数据
            user_friends_df: 用户-用户好友关系数据
            artists_df: 艺术家数据（仅包含id和name）
        """
        # 加载用户-艺术家关系数据
        user_artists_df = pd.read_csv(f"{self.data_path}user_artists.dat", sep='\t', engine='python',
                                      names=['userId', 'artistId', 'weight'])
        user_artists_df['weight'] = pd.to_numeric(user_artists_df['weight'], errors='coerce')
        user_artists_df['itemId'] = user_artists_df['artistId']

        # 加载用户-用户好友关系数据
        user_friends_df = pd.read_csv(f"{self.data_path}user_friends.dat", sep='\t', engine='python',
                                      names=['userId', 'friendId'])

        # 加载艺术家数据，只选择id和name列
        artists_df = pd.read_csv(f"{self.data_path}artists.dat", sep='\t', engine='python', quoting=csv.QUOTE_NONE,
                                 usecols=[0, 1], names=['artistId', 'name'])


        user_artists_df = user_artists_df.iloc[1:]
        user_friends_df = user_friends_df.iloc[1:]
        artists_df = artists_df.iloc[1:]

        # 按用户ID分组，计算每个用户的艺术家数量和总播放次数
        user_stats = user_artists_df.groupby('userId').agg(
            artist_count=('artistId', 'nunique'),
            total_weight=('weight', 'sum')
        ).reset_index()

        # 过滤掉不符合条件的用户
        valid_users = user_stats[
            (user_stats['artist_count'] >= 10) &
            (user_stats['total_weight'] >= 50)
            ]['userId']

        # 筛选符合条件的用户的数据
        filtered_df = user_artists_df[user_artists_df['userId'].isin(valid_users)]

        # 按用户ID分组，计算每个用户的播放次数的最大值和最小值
        user_max_min = filtered_df.groupby('userId')['weight'].agg(['max', 'min']).reset_index()

        # 合并最大值和最小值到原始数据
        rated_df = pd.merge(filtered_df, user_max_min, on='userId')

        rated_df['rating'] = rated_df.apply(
            lambda row: (
                    (np.log10(row['weight'] + 1) / np.log10(row['max'] + 1)) * 5.0
            ) if row['max'] != row['min'] else 4.0,
            axis=1
        )

        # 确保评分在1.0到5.0之间
        rated_df['rating'] = rated_df['rating'].clip(1.0, 5.0)

        # 删除中间列
        rated_df.drop(['max', 'min'], axis=1, inplace=True)

        print(
            f"Loaded user_artists: {len(rated_df)}, user_friends: {len(user_friends_df)}, artists: {len(artists_df)}")

        return rated_df, user_friends_df, artists_df

    def load_yelp_data(self):
        """
        加载 Yelp 数据集（review.csv, user.csv, business.csv）

        Returns:
            ratings_df: 包含 userId, itemId, rating, timestamp（可选）
            users_df: 原始用户信息
            business_df: 原始商家信息
        """

        # 1. 加载基础表格
        reviews = pd.read_csv(f"{self.data_path}review.csv")
        users = pd.read_csv(f"{self.data_path}user.csv")
        businesses = pd.read_csv(f"{self.data_path}business.csv")

        # 2. 清洗 & 重命名字段统一格式（兼容已有框架）
        ratings_df = reviews.rename(columns={
            'user_id': 'userId',
            'business_id': 'itemId',
            'stars': 'rating',
            'date': 'timestamp'
        })[['userId', 'itemId', 'rating', 'timestamp']]

        # 3. 过滤活跃用户（如评论数 ≥10）
        active_users = users[users['review_count'] >= 10]['user_id']
        ratings_df = ratings_df[ratings_df['userId'].isin(active_users)]

        # 4. 类型转换（保证兼容性）
        ratings_df['rating'] = ratings_df['rating'].astype(float)
        ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'])

        # 5. 返回（users 和 businesses 原样保留）
        print(f" Loaded Yelp: {len(ratings_df)} ratings from {ratings_df['userId'].nunique()} users")

        return ratings_df, users, businesses

    def preprocess(self, ratings_df):  
        """预处理评分数据  
        
        Args:  
            ratings_df: 原始评分数据  

        Returns:  
            处理后的评分数据  
        """  
        # 这里可以添加更多预处理步骤，如去除评分过少的用户和电影  
        # 目前只保留评分 >= 4 的作为正例  
        processed_df = ratings_df.copy()  
        processed_df['liked'] = (processed_df['rating'] >= 3.7).astype(int)  
        
        # 确保用户ID和电影ID是连续的整数  
        user_ids = processed_df['userId'].unique()  
        item_ids = processed_df['itemId'].unique()  
        
        user_map = {old: new for new, old in enumerate(user_ids)}  
        item_map = {old: new for new, old in enumerate(item_ids)}  
        
        processed_df['user_idx'] = processed_df['userId'].map(user_map)  
        processed_df['item_idx'] = processed_df['itemId'].map(item_map)

        # processed_df = processed_df.sample(frac=0.1, random_state=42)
        
        return processed_df
    
    def create_user_item_matrix(self, ratings_df):  
        """创建用户-物品评分矩阵  
        
        Args:  
            ratings_df: 预处理后的评分数据  
            
        Returns:  
            用户-物品评分矩阵  
        """  
        n_users = ratings_df['user_idx'].nunique()  
        n_items = ratings_df['item_idx'].nunique()  
        
        # 创建矩阵  
        matrix = np.zeros((n_users, n_items))  
        
        for _, row in ratings_df.iterrows():  
            matrix[int(row['user_idx']), int(row['item_idx'])] = row['rating']  
            
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
            # liked_items = user_ratings.copy()
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

    def split_data_by_time(self, ratings_df, test_size=0.2):
        """根据时间戳划分训练集和测试集

        Args:
            ratings_df: 预处理后的评分数据
            test_size: 测试集比例

        Returns:
            train_df: 训练集
            test_df: 测试集
        """
        # 按时间戳排序
        liked_items = ratings_df[ratings_df['liked'] == 1]
        # liked_items = ratings_df.copy()
        liked_items = liked_items.sort_values('timestamp')

        # 计算划分点
        split_index = int(len(liked_items) * (1 - test_size))

        # 划分训练集和测试集
        train_df = liked_items.iloc[:split_index]
        test_df = liked_items.iloc[split_index:]

        print(f"Training set: {len(train_df)}, Test set: {len(test_df)}")
        return train_df, test_df
    
    def split_data_random(self, ratings_df, test_size=0.2, random_state=42):
        """随机划分训练集和测试集

        Args:
            ratings_df: 预处理后的评分数据
            test_size: 测试集比例
            random_state: 随机种子

        Returns:
            train_df: 训练集
            test_df: 测试集
        """
        # 随机划分数据集
        liked_items = ratings_df[ratings_df['liked'] == 1]
        # liked_items = ratings_df.copy()
        train_df, test_df = train_test_split(
            liked_items,
            test_size=test_size,
            random_state=random_state
        )

        print(f"Training set: {len(train_df)}, Test set: {len(test_df)}")
        return train_df, test_df