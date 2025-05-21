# CS5481-RecommenderSystem
# Recommendation System Code Repository

This code repository contains implementations of various recommendation algorithms and evaluation tools, supporting multiple datasets and evaluation metrics.

## Directory Structure

├── base_recommender.py          # Base class for all recommender algorithms

├── models/

│   ├── user_cf.py               # User-based Collaborative Filtering

│   ├── itemcf.py                # Item-based Collaborative Filtering

│   ├── baseline.py              # Baseline models (Random and Popularity recommenders)

│   ├── content_based.py         # Content-based Recommendation

│   ├── friend_based.py          # Friend-based Collaborative Filtering

│   ├── SVD.py                   # Matrix Factorization with SVD

│   ├── NGCF.py                  # Neural Graph Collaborative Filtering

│   ├── LGCN.py                  # LightGCN Graph Neural Network

│   └── init.py

├── data_processor.py            # Data processing utilities

├── evaluation.py                # Evaluation metrics and tools

└── experiment.py                # Main experiment script


## Code File Descriptions

### `base_recommender.py`
Defines the base class for all recommender algorithms, providing a unified interface with `fit` and `recommend` methods.

### `models/user_cf.py`
Implements user-based collaborative filtering, recommending items by calculating user similarities.

### `models/itemcf.py`
Implements item-based collaborative filtering, recommending items by calculating item similarities.

### `models/baseline.py`
Contains two baseline recommendation algorithms:
- `RandomRecommender`: Recommends items randomly.
- `PopularityRecommender`: Recommends items based on their popularity.

### `models/content_based.py`
Content-based recommendation algorithm that uses item feature information (e.g., movie genres) for recommendations.

### `models/friend_based.py`
Collaborative filtering algorithm that incorporates user friendship relationships for recommendations.

### `models/SVD.py`
Matrix factorization-based recommendation algorithm using Singular Value Decomposition (SVD).

### `models/NGCF.py`
Neural Graph Collaborative Filtering algorithm that captures high-order user-item relationships using graph neural networks.

### `models/LGCN.py`
LightGCN algorithm, an efficient graph neural network-based recommendation method.

### `data_processor.py`
Utilities for data loading, preprocessing, and splitting, supporting multiple datasets.

### `evaluation.py`
Tools for evaluating recommender systems, providing multiple metrics (e.g., HR@K, NDCG@K, MRR@K, Recall@K).

### `experiment.py`
Main script for running experiments with different recommendation algorithms.

## Supported Datasets

- MovieLens-1M (`ml-1m`)
- Last.fm (`lastfm`)
- Yelp (`yelp`)

## Experiment Command-Line Scripts

### MovieLens-1M Dataset

#### UserCF Model
```bash
python experiment.py --dataset ml-1m --data_path /path/to/ml-1m/ --model usercf --k 20 --split_method default
python experiment.py --dataset ml-1m --data_path /path/to/ml-1m/ --model usercf --k 30 --split_method default
python experiment.py --dataset ml-1m --data_path /path/to/ml-1m/ --model usercf --k 40 --split_method default

#### Data Splitting Methods
python experiment.py --dataset ml-1m --data_path /path/to/ml-1m/ --model usercf --k 30 --split_method time
python experiment.py --dataset ml-1m --data_path /path/to/ml-1m/ --model usercf --k 30 --split_method random

#### Other Models
python experiment.py --dataset ml-1m --data_path /path/to/ml-1m/ --model svd --n_factors 500 --n_epochs 300 --split_method default
python experiment.py --dataset ml-1m --data_path /path/to/ml-1m/ --model lightgcn --embed_dim 256 --n_layers 4 --epochs 5 --split_method default
python experiment.py --dataset ml-1m --data_path /path/to/ml-1m/ --model ngcf --embed_dim 64 --n_layers 3 --epochs 5 --split_method default
python experiment.py --dataset ml-1m --data_path /path/to/ml-1m/ --model random --split_method default
python experiment.py --dataset ml-1m --data_path /path/to/ml-1m/ --model popular --split_method default
python experiment.py --dataset ml-1m --data_path /path/to/ml-1m/ --model content --split_method default

#### Last.fm Dataset
python experiment.py --dataset lastfm --data_path /path/to/lastfm/ --model usercf --k 20 --split_method default
python experiment.py --dataset lastfm --data_path /path/to/lastfm/ --model itemcf --k 20 --split_method default
python experiment.py --dataset lastfm --data_path /path/to/lastfm/ --model svd --n_factors 200 --n_epochs 500 --split_method default
python experiment.py --dataset lastfm --data_path /path/to/lastfm/ --model lightgcn --embed_dim 128 --n_layers 2 --epochs 20 --split_method default
python experiment.py --dataset lastfm --data_path /path/to/lastfm/ --model ngcf --embed_dim 64 --n_layers 3 --epochs 20 --split_method default
python experiment.py --dataset lastfm --data_path /path/to/lastfm/ --model random --split_method default
python experiment.py --dataset lastfm --data_path /path/to/lastfm/ --model popular --split_method default

#### Yelp Dataset
python experiment.py --dataset yelp --data_path /path/to/yelp/ --model usercf --k 20 --split_method default
python experiment.py --dataset yelp --data_path /path/to/yelp/ --model svd --n_factors 200 --n_epochs 500 --split_method default
python experiment.py --dataset yelp --data_path /path/to/yelp/ --model lightgcn --embed_dim 128 --n_layers 2 --epochs 20 --split_method default
python experiment.py --dataset yelp --data_path /path/to/yelp/ --model ngcf --embed_dim 64 --n_layers 3 --epochs 20 --split_method default
python experiment.py --dataset yelp --data_path /path/to/yelp/ --model random --split_method default
python experiment.py --dataset yelp --data_path /path/to/yelp/ --model popular --split_method default

### Notes
The MovieLens dataset supports the time splitting method, while Last.fm and Yelp datasets do not.
Ensure all dependencies are installed and adjust data paths as needed.
