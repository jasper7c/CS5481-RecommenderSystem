import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.sparse import FloatTensor as SparseTensor
from .base_recommender import BaseRecommender
from .baseline import PopularityRecommender

class LightGCN(BaseRecommender):
    """基于LightGCN的推荐算法，使用BPR损失"""
    
    def __init__(self, embed_dim=64, n_layers=4, lr=0.001, epochs=100, batch_size=1024, weight_decay=1e-4):
        """初始化LightGCN推荐器
        
        Args:
            embed_dim (int): 嵌入维度
            n_layers (int): 图卷积层数
            lr (float): 学习率
            epochs (int): 训练轮数
            batch_size (int): 批大小
            weight_decay (float): L2正则化系数
        """
        super().__init__("LightGCN")
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        # 以下属性在fit时初始化
        self.user_mappings = None
        self.item_mappings = None
        self.n_users = 0
        self.n_items = 0
        self.adj_matrix = None          # 归一化后的邻接矩阵（稀疏张量）
        self.embedding = None           # 用户和物品的嵌入
        self.user_pos_items = None      # 记录每个用户的交互物品
        
        self.coldstart = PopularityRecommender()

    def fit(self, train_data):
        """训练模型
        
        Args:
            train_data: 训练数据，包含userId和itemId字段
        """
        # 创建用户和物品的映射
        self.user_mappings = {
            'idx_to_id': dict(enumerate(train_data['userId'].unique())),
            'id_to_idx': {id: idx for idx, id in enumerate(train_data['userId'].unique())}
        }
        self.item_mappings = {
            'idx_to_id': dict(enumerate(train_data['itemId'].unique())),
            'id_to_idx': {id: idx for idx, id in enumerate(train_data['itemId'].unique())}
        }
        self.n_users = len(self.user_mappings['id_to_idx'])
        self.n_items = len(self.item_mappings['id_to_idx'])
        
        self.coldstart.fit(train_data)
        
        # 记录用户的交互物品（用于负采样和排除已知物品）
        self.user_pos_items = defaultdict(set)
        user_indices = train_data['userId'].map(self.user_mappings['id_to_idx']).tolist()
        item_indices = train_data['itemId'].map(self.item_mappings['id_to_idx']).tolist()
        for u, i in zip(user_indices, item_indices):
            self.user_pos_items[u].add(i)
        
        # 构建归一化的邻接矩阵（稀疏张量）
        self._build_adjacency_matrix(user_indices, item_indices)
        
        # 初始化嵌入
        self.embedding = nn.Embedding(self.n_users + self.n_items, self.embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        
        # 转换为GPU（如果可用）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.adj_matrix = self.adj_matrix.to(self.device)
        self.embedding = self.embedding.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            [self.embedding.weight], 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # 训练循环
        print("Training LightGCN...")
        for epoch in range(self.epochs):
            self._train_one_epoch(epoch, user_indices, item_indices)
            
        # 计算最终的嵌入（各层平均）
        with torch.no_grad():
            final_emb = self._compute_final_embeddings()
            self.user_emb = final_emb[:self.n_users]
            self.item_emb = final_emb[self.n_users:]

    def _build_adjacency_matrix(self, user_indices, item_indices):
        """构建归一化的邻接矩阵（稀疏张量）"""
        # 构建用户-物品交互矩阵
        rows = torch.LongTensor(user_indices)
        cols = torch.LongTensor(item_indices) + self.n_users  # 物品节点在邻接矩阵中的位置
        values = torch.ones(len(rows))
        
        # 创建对称矩阵（用户-物品和物品-用户边）
        adj_rows = torch.cat([rows, cols])
        adj_cols = torch.cat([cols, rows])
        adj_values = torch.cat([values, values])
        
        # 计算归一化系数
        node_degrees = torch.zeros(self.n_users + self.n_items)
        for u, i in zip(user_indices, item_indices):
            node_degrees[u] += 1
            node_degrees[self.n_users + i] += 1
        norm_values = 1.0 / torch.sqrt(node_degrees[adj_rows] * node_degrees[adj_cols])
        
        # 创建稀疏张量
        self.adj_matrix = SparseTensor(
            torch.stack([adj_rows, adj_cols]), 
            norm_values,
            torch.Size([self.n_users + self.n_items, self.n_users + self.n_items])
        )

    def _compute_final_embeddings(self):
        """计算各层嵌入的平均"""
        embeds = [self.embedding.weight]
        for _ in range(self.n_layers):
            embeds.append(torch.sparse.mm(self.adj_matrix, embeds[-1]))
        return torch.mean(torch.stack(embeds), dim=0)

    def _train_one_epoch(self, epoch, user_indices, item_indices):
        """单轮训练"""
        total_loss = 0.0
        all_pos_pairs = list(zip(user_indices, item_indices))
        np.random.shuffle(all_pos_pairs)
        
        for idx in range(0, len(all_pos_pairs), self.batch_size):
            batch_pairs = all_pos_pairs[idx: idx+self.batch_size]
            users = torch.LongTensor([u for u, _ in batch_pairs]).to(self.device)
            pos_items = torch.LongTensor([i for _, i in batch_pairs]).to(self.device)
            
            # 采样负物品
            neg_items = []
            for u, _ in batch_pairs:
                while True:
                    neg_i = np.random.randint(self.n_items)
                    if neg_i not in self.user_pos_items[u]:
                        neg_items.append(neg_i)
                        break
            neg_items = torch.LongTensor(neg_items).to(self.device)
            
            # 计算嵌入
            final_emb = self._compute_final_embeddings()
            user_emb = final_emb[users]
            pos_emb = final_emb[self.n_users + pos_items]
            neg_emb = final_emb[self.n_users + neg_items]
            
            # BPR损失计算
            pos_scores = (user_emb * pos_emb).sum(dim=1)
            neg_scores = (user_emb * neg_emb).sum(dim=1)
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(all_pos_pairs):.4f}")

    def recommend(self, user_id, n_recommendations=10, exclude_known=True):
        """生成推荐列表"""
        if user_id not in self.user_mappings['id_to_idx']:
            return self.coldstart.recommend(user_id, n_recommendations)
        
        user_idx = self.user_mappings['id_to_idx'][user_id]
        user_tensor = torch.LongTensor([user_idx]).to(self.device)
        user_emb = self.user_emb[user_tensor]
        scores = torch.matmul(user_emb, self.item_emb.T).cpu().detach().numpy().flatten()
        
        # 排除已知物品
        if exclude_known:
            interacted = list(self.user_pos_items.get(user_idx, []))
            scores[interacted] = -np.inf
        
        # 获取Top-K推荐
        top_indices = np.argsort(scores)[-n_recommendations:][::-1]
        return [self.item_mappings['idx_to_id'][idx] for idx in top_indices if scores[idx] > -np.inf]