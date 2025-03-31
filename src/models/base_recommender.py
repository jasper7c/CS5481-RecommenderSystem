class BaseRecommender:  
    """所有推荐算法的基类，定义了共同的接口"""  
    
    def __init__(self, name):  
        """初始化推荐器  
        
        Args:  
            name: 推荐器名称  
        """  
        self.name = name  
        
    def fit(self, train_data):  
        """训练模型  
        
        Args:  
            train_data: 训练数据  
        """  
        raise NotImplementedError("子类必须实现fit方法")  
        
    def recommend(self, user_id, n_recommendations=10, exclude_known=True):  
        """为用户生成推荐  
        
        Args:  
            user_id: 用户ID  
            n_recommendations: 推荐数量  
            exclude_known: 是否排除已知项目  
            
        Returns:  
            推荐的物品ID列表  
        """  
        raise NotImplementedError("子类必须实现recommend方法")  