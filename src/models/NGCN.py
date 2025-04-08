import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.sparse import FloatTensor as SparseTensor
from .base_recommender import BaseRecommender
from .baseline import PopularityRecommender

class NGCNRecommender(BaseRecommender, nn.Module):
    """基于神经图协同过滤的推荐算法"""
    
    def __init__(self, embed_dim=64, n_layers=3, lr=0.001, 
                 epochs=100, batch_size=1024, weight_decay=1e-4, 
                 dropout=0.2, use_residual=True):
        """
        Args:
            embed_dim: 嵌入维度
            n_layers: 图卷积层数
            lr: 学习率
            epochs: 训练轮数
            batch_size: 批大小
            weight_decay: L2正则化系数
            dropout: Dropout概率
            use_residual: 是否使用残差连接
        """
        BaseRecommender.__init__(self, name="NGCN")
        nn.Module.__init__(self)  # 关键：初始化PyTorch模块
        
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.use_residual = use_residual

        # 运行时初始化的属性
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.user_mappings = None
        self.item_mappings = None
        self.n_users = 0
        self.n_items = 0
        self.adj_matrix = None
        self.user_pos_items = None
        
        self.coldstart = PopularityRecommender()

    def fit(self, train_data):
        """训练模型"""
        # 初始化数据映射
        self._init_mappings(train_data)
        self.n_users = len(self.user_mappings['idx_to_id'])
        self.n_items = len(self.item_mappings['idx_to_id'])
        
        self.coldstart.fit(train_data)
        
        # 构建邻接矩阵
        self._build_adjacency_matrix(train_data)
        
        # 初始化模型组件
        self._init_model_components()
        
        # 训练循环
        self._train_model(train_data)
        

    def _init_model_components(self):
        """初始化模型参数和组件"""
        # 基础嵌入层
        self.base_embedding = nn.Embedding(
            num_embeddings=self.n_users + self.n_items,
            embedding_dim=self.embed_dim
        ).to(self.device)
        nn.init.xavier_normal_(self.base_embedding.weight)
        
        # 图卷积层
        self.gcn_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            layer = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            nn.init.xavier_normal_(layer.weight)
            self.gcn_layers.append(layer.to(self.device))
        
        # Dropout层
        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        # 组合层
        self.combine_layer = nn.Linear(
            self.embed_dim * (self.n_layers + 1), 
            self.embed_dim
        ).to(self.device)
        nn.init.xavier_normal_(self.combine_layer.weight)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def _build_adjacency_matrix(self, data):
        """改进后的邻接矩阵构建（去重+向量化计算）"""
        # 获取去重后的交互数据
        unique_data = data[['userId', 'itemId']].drop_duplicates()
        user_indices = unique_data['userId'].map(self.user_mappings['id_to_idx']).tolist()
        item_indices = unique_data['itemId'].map(self.item_mappings['id_to_idx']).tolist()

        # 向量化计算度数
        u_indices = np.array(user_indices)
        i_indices = np.array(item_indices)
        
        node_degrees = torch.zeros(self.n_users + self.n_items)
        node_degrees[:self.n_users] = torch.from_numpy(np.bincount(u_indices, minlength=self.n_users)).float()
        node_degrees[self.n_users:] = torch.from_numpy(np.bincount(i_indices, minlength=self.n_items)).float()
        node_degrees = torch.clamp(node_degrees, min=1e-12)

        # 构建对称连接
        rows = torch.LongTensor(user_indices + (np.array(item_indices) + self.n_users).tolist())
        cols = torch.LongTensor((np.array(item_indices) + self.n_users).tolist() + user_indices)
        
        # 计算归一化系数
        norm = 1.0 / torch.sqrt(node_degrees[rows] * node_degrees[cols])
        
        # 创建稀疏张量
        self.adj_matrix = SparseTensor(
            torch.stack([rows, cols]), 
            norm,
            torch.Size([self.n_users + self.n_items, self.n_users + self.n_items])
        ).to(self.device)
        
    def forward(self):
        """前向传播生成组合嵌入"""
        embeddings = [self.base_embedding.weight]
        current_emb = self.base_embedding.weight
        
        for i, layer in enumerate(self.gcn_layers):
            # 图卷积操作
            new_emb = torch.sparse.mm(self.adj_matrix, current_emb)
            
            # 特征变换和非线性激活
            new_emb = layer(new_emb)
            new_emb = F.relu(new_emb)
            new_emb = self.dropout_layer(new_emb)
            
            # 残差连接
            if self.use_residual and i > 0:
                new_emb += embeddings[-1]
            
            embeddings.append(new_emb)
            current_emb = new_emb
        
        # 拼接各层嵌入
        combined_emb = torch.cat(embeddings, dim=1)
        
        # 组合层
        final_emb = self.combine_layer(combined_emb)
        return final_emb

    def _train_model(self, data):
        """训练主循环"""
        user_indices = data['userId'].map(self.user_mappings['id_to_idx']).tolist()
        item_indices = data['itemId'].map(self.item_mappings['id_to_idx']).tolist()
        self.user_pos_items = defaultdict(set)
        
        for u, i in zip(user_indices, item_indices):
            self.user_pos_items[u].add(i)

        print("Training NGCN...")
        for epoch in range(self.epochs):
            total_loss = 0.0
            all_pairs = list(zip(user_indices, item_indices))
            np.random.shuffle(all_pairs)
            
            for idx in range(0, len(all_pairs), self.batch_size):
                batch = all_pairs[idx: idx + self.batch_size]
                loss = self._train_batch(batch)
                total_loss += loss.item()
            
            avg_loss = total_loss / len(all_pairs)
            print(f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f}")

    def _train_batch(self, batch):
        """单批训练"""
        # 生成负样本
        users, pos_items, neg_items = [], [], []
        for u, p in batch:
            users.append(u)
            pos_items.append(p)
            while True:
                n = np.random.randint(self.n_items)
                if n not in self.user_pos_items[u]:
                    neg_items.append(n)
                    break
        
        # 转换为Tensor
        users = torch.LongTensor(users).to(self.device)
        pos_items = torch.LongTensor(pos_items).to(self.device) + self.n_users
        neg_items = torch.LongTensor(neg_items).to(self.device) + self.n_users
        
        # 前向传播
        all_emb = self.forward()
        user_emb = all_emb[users]
        pos_emb = all_emb[pos_items]
        neg_emb = all_emb[neg_items]
        
        # 计算BPR损失
        pos_scores = (user_emb * pos_emb).sum(dim=1)
        neg_scores = (user_emb * neg_emb).sum(dim=1)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss

    def recommend(self, user_id, n_recommendations=10, exclude_known=True):
        """生成推荐列表"""
        if user_id not in self.user_mappings['id_to_idx']:
            return self.coldstart.recommend(user_id, n_recommendations)
        
        user_idx = self.user_mappings['id_to_idx'][user_id]
        
        with torch.no_grad():  # 禁用梯度计算
            final_emb = self.forward()
        
        # 分离用户和物品嵌入
        user_emb = final_emb[:self.n_users]
        item_emb = final_emb[self.n_users:]
        
        # 计算得分
        scores = torch.mm(user_emb[user_idx].unsqueeze(0), item_emb.T).squeeze(0)
        scores = scores.cpu().detach().numpy()
        
        # 排除已知物品
        if exclude_known:
            rated = list(self.user_pos_items.get(user_idx, []))
            scores[rated] = -np.inf
        
        # 获取Top-K推荐
        top_indices = np.argsort(scores)[-n_recommendations:][::-1]
        return [self.item_mappings['idx_to_id'][idx] for idx in top_indices if scores[idx] > -np.inf]

    def _init_mappings(self, data):
        """初始化用户/物品映射"""
        self.user_mappings = {
            'idx_to_id': dict(enumerate(data['userId'].unique())),
            'id_to_idx': {id: idx for idx, id in enumerate(data['userId'].unique())}
        }
        self.item_mappings = {
            'idx_to_id': dict(enumerate(data['itemId'].unique())),
            'id_to_idx': {id: idx for idx, id in enumerate(data['itemId'].unique())}
        }