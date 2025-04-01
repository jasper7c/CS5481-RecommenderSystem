import numpy as np  
import pandas as pd  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity  
from .base_recommender import BaseRecommender  

# 尝试导入GPU加速库  
try:  
    import cupy as cp  
    from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix  
    from cupy.linalg import norm as cp_norm  
    GPU_AVAILABLE = True  
except ImportError:  
    cp = np  
    GPU_AVAILABLE = False  

class ContentBased(BaseRecommender):  
    """基于内容的推荐算法，使用电影类型信息"""  
    
    def __init__(self):  
        """初始化基于内容的推荐器"""  
        super().__init__("ContentBased")  
        self.tfidf_matrix = None  
        self.user_profiles = {}  
        self.movies_df = None  
        self.movie_indices = {}  
        self.movie_ids = []  
        self.use_gpu = GPU_AVAILABLE  
        
    def fit(self, train_data, movies_df):  
        """训练模型  
        
        Args:  
            train_data: 训练数据，包含userId, movieId, rating字段  
            movies_df: 电影数据，包含movieId, title, genres字段  
        """  
        print(f"Using GPU acceleration: {self.use_gpu}")  
        
        self.movies_df = movies_df.copy()  
        
        # 预处理电影特征  
        self._preprocess_movie_features()  
        
        # 创建用户画像(profile)  
        self._create_user_profiles(train_data)  
        
        print(f"Content-Based model trained with {len(self.movies_df)} movies and {len(self.user_profiles)} user profiles")  
        
    def _preprocess_movie_features(self):  
        """预处理电影特征，将类型转换为TF-IDF向量"""  
        # 确保genres是字符串格式  
        self.movies_df['genres'] = self.movies_df['genres'].fillna('')
        self.movies_df['title'] = self.movies_df['title'].fillna('')
        
        # 替换分隔符以便TF-IDF处理  
        self.movies_df['genres_str'] = self.movies_df['genres'].str.replace('|', ' ')

        # 合并标题和类型信息
        self.movies_df['combined'] = self.movies_df['title'] + " " + self.movies_df['genres_str']
        
        # # 使用TF-IDF向量化电影类型
        # print("Vectorizing movie genres...")
        # tfidf = TfidfVectorizer(stop_words='english')
        # self.tfidf_matrix = tfidf.fit_transform(self.movies_df['genres_str'])

        # 使用TF-IDF向量化合并后的特征
        print("Vectorizing combined features...")
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['combined'])
        
        # 如果GPU可用，转移到GPU  
        if self.use_gpu:  
            try:  
                # 转换为CuPy CSR矩阵  
                self.tfidf_matrix = cp_csr_matrix(self.tfidf_matrix)  
                print("TF-IDF matrix transferred to GPU")  
            except Exception as e:  
                print(f"Error transferring to GPU: {e}")  
                self.use_gpu = False  
        
        # 创建电影ID到索引的映射  
        self.movie_indices = {movie_id: idx for idx, movie_id in enumerate(self.movies_df['movieId'].values)}  
        self.movie_ids = self.movies_df['movieId'].values  
        
    def _create_user_profiles(self, train_data):  
        """为每个用户创建内容画像，基于他们评价过的电影"""  
        print("Creating user profiles...")  
        
        # 按用户分组  
        user_groups = train_data.groupby('userId')  
        
        for user_id, group in user_groups:  
            # 获取用户评价过的电影  
            user_movies = pd.merge(group, self.movies_df, on='movieId')  
            
            # 只考虑评分较高的电影(>=3.5)作为用户喜好  
            liked_movies = user_movies[user_movies['rating'] >= 3.5]  
            
            if len(liked_movies) == 0:  
                continue  
                
            # 获取用户喜欢的电影的特征向量  
            user_profile = np.zeros(self.tfidf_matrix.shape[1])  
            
            for _, movie in liked_movies.iterrows():  
                if movie['movieId'] in self.movie_indices:  
                    idx = self.movie_indices[movie['movieId']]  
                    if self.use_gpu:  
                        # 对于GPU，需要将稀疏矩阵转为密集矩阵  
                        movie_vec = self.tfidf_matrix[idx].toarray().flatten()  
                        # 转回CPU进行累加  
                        movie_vec = cp.asnumpy(movie_vec)  
                    else:  
                        movie_vec = self.tfidf_matrix[idx].toarray().flatten()  
                    
                    # 加权累加，评分越高权重越大  
                    user_profile += movie_vec * (movie['rating'] / 5.0)  
            
            # 归一化用户画像  
            norm = np.linalg.norm(user_profile)  
            if norm > 0:  
                user_profile = user_profile / norm  
                
            self.user_profiles[user_id] = user_profile  
    
    def recommend(self, user_id, n_recommendations=10, exclude_known=True):  
        """为用户生成推荐  
        
        Args:  
            user_id: 用户ID  
            n_recommendations: 推荐数量  
            exclude_known: 是否排除已知物品  
            
        Returns:  
            推荐的物品ID列表  
        """  
        if user_id not in self.user_profiles:  
            return []  
            
        # 获取用户画像  
        user_profile = self.user_profiles[user_id]  
        
        # 计算用户画像与所有电影特征的相似度  
        if self.use_gpu:  
            # 转移用户画像到GPU  
            user_profile_gpu = cp.array(user_profile)  
            
            # 计算相似度  
            # 对于大型矩阵，分批处理以节省内存  
            batch_size = 1000  
            n_batches = (self.tfidf_matrix.shape[0] + batch_size - 1) // batch_size  
            
            similarities = np.zeros(self.tfidf_matrix.shape[0])  
            
            for i in range(n_batches):  
                start_idx = i * batch_size  
                end_idx = min((i+1) * batch_size, self.tfidf_matrix.shape[0])  
                
                batch = self.tfidf_matrix[start_idx:end_idx].toarray()  
                batch_gpu = cp.array(batch)  
                
                # 计算余弦相似度  
                batch_sim = cp.dot(batch_gpu, user_profile_gpu) / (  
                    cp_norm(batch_gpu, axis=1) * cp_norm(user_profile_gpu) + 1e-10  
                )  
                
                # 转回CPU  
                similarities[start_idx:end_idx] = cp.asnumpy(batch_sim)  
        else:  
            # CPU版本：一次性计算所有相似度  
            # 将矩阵转为密集矩阵，可能会消耗大量内存  
            similarities = cosine_similarity(  
                self.tfidf_matrix.toarray(),   
                user_profile.reshape(1, -1)  
            ).flatten()  
        
        # 获取排名前N的电影索引  
        movie_indices = np.argsort(similarities)[::-1]  
        
        # 如果需要排除已知电影  
        if exclude_known:  
            # 获取用户已评价过的电影  
            rated_movies = [movie_id for movie_id, profile in self.user_profiles.items() if np.array_equal(profile, user_profile)]  
            
            # 过滤掉已评价的电影  
            movie_indices = [idx for idx in movie_indices if self.movie_ids[idx] not in rated_movies]  
        
        # 选择前N个推荐  
        top_movie_indices = movie_indices[:n_recommendations]  
        
        # 转换为电影ID  
        recommendations = [self.movie_ids[idx] for idx in top_movie_indices]  
        
        return recommendations  
    
    def get_similar_movies(self, movie_id, n=10):  
        """获取与给定电影相似的电影  
        
        Args:  
            movie_id: 电影ID  
            n: 相似电影数量  
            
        Returns:  
            相似电影ID列表  
        """  
        if movie_id not in self.movie_indices:  
            return []  
            
        idx = self.movie_indices[movie_id]  
        
        # 获取电影特征向量  
        if self.use_gpu:  
            # 获取特定电影的特征向量  
            movie_vec = self.tfidf_matrix[idx].toarray().flatten()  
            movie_vec_gpu = cp.array(movie_vec)  
            
            # 计算相似度 (分批处理)  
            batch_size = 1000  
            n_batches = (self.tfidf_matrix.shape[0] + batch_size - 1) // batch_size  
            
            similarities = np.zeros(self.tfidf_matrix.shape[0])  
            
            for i in range(n_batches):  
                start_idx = i * batch_size  
                end_idx = min((i+1) * batch_size, self.tfidf_matrix.shape[0])  
                
                batch = self.tfidf_matrix[start_idx:end_idx].toarray()  
                batch_gpu = cp.array(batch)  
                
                # 计算余弦相似度  
                batch_sim = cp.dot(batch_gpu, movie_vec_gpu) / (  
                    cp_norm(batch_gpu, axis=1) * cp_norm(movie_vec_gpu) + 1e-10  
                )  
                
                # 转回CPU  
                similarities[start_idx:end_idx] = cp.asnumpy(batch_sim)  
        else:  
            # CPU版本  
            movie_vec = self.tfidf_matrix[idx].toarray().reshape(1, -1)  
            similarities = cosine_similarity(self.tfidf_matrix, movie_vec).flatten()  
        
        # 获取排名前N的电影索引 (排除自身)  
        similar_indices = np.argsort(similarities)[::-1][1:n+1]  
        
        # 转换为电影ID  
        similar_movies = [self.movie_ids[i] for i in similar_indices]  
        
        return similar_movies  