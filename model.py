import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import pickle

class RecommendationModel:
    def __init__(self, n_components=100):
        self.model = TruncatedSVD(n_components=n_components)
        self.user_features = None
        self.item_features = None

    def fit(self, user_item_matrix):
        self.user_features = self.model.fit_transform(user_item_matrix)
        self.item_features = self.model.components_.T
        self.user_features = normalize(self.user_features)
        self.item_features = normalize(self.item_features)

    def recommend(self, user_id, n_recommendations=5):
        user_vector = self.user_features[user_id]
        scores = np.dot(self.item_features, user_vector)
        top_items = np.argsort(scores)[::-1][:n_recommendations]
        return top_items.tolist()

    def update(self, user_id, item_id, interaction_value):
        self.user_features[user_id] += self.item_features[item_id] * interaction_value
        self.user_features[user_id] = normalize(self.user_features[user_id].reshape(1, -1))[0]

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def create_sample_data():
    products = {
        'product_id': range(1, 21),
        'name': [
            'Wireless Headphones', 'Smartphone', 'Laptop', 'Smartwatch', 'Tablet',
            'Gaming Console', 'Digital Camera', 'Bluetooth Speaker', 'Fitness Tracker', 'E-reader',
            'Power Bank', 'Wireless Charger', 'Smart TV', 'Gaming Mouse', 'Mechanical Keyboard',
            'Monitor', 'Graphics Card', 'RAM', 'SSD', 'CPU'
        ],
        'category': [
            'Audio', 'Phones', 'Computers', 'Wearables', 'Tablets',
            'Gaming', 'Cameras', 'Audio', 'Wearables', 'Electronics',
            'Accessories', 'Accessories', 'Electronics', 'Gaming', 'Gaming',
            'Computers', 'Computers', 'Computers', 'Computers', 'Computers'
        ],
        'price': [
            199.99, 899.99, 1299.99, 299.99, 499.99,
            499.99, 699.99, 129.99, 99.99, 129.99,
            49.99, 39.99, 799.99, 79.99, 129.99,
            299.99, 699.99, 89.99, 149.99, 399.99
        ],
        'description': [
            'High-quality wireless headphones with noise cancellation',
            'Latest smartphone with advanced camera system',
            'Powerful laptop for work and gaming',
            'Smart watch with fitness tracking and notifications',
            'Versatile tablet for entertainment and productivity',
            'Next-gen gaming console for immersive gaming',
            'Professional digital camera with 4K video',
            'Portable bluetooth speaker with rich sound',
            'Advanced fitness tracker with heart rate monitoring',
            'E-reader with paper-like display',
            'High-capacity portable power bank',
            'Fast wireless charger for smartphones',
            '4K Smart TV with streaming apps',
            'Precise gaming mouse with RGB lighting',
            'Mechanical gaming keyboard with customizable keys',
            'High refresh rate gaming monitor',
            'High-performance graphics card',
            'Fast DDR4 RAM module',
            'Fast NVMe SSD storage',
            'High-performance CPU processor'
        ]
    }
    return pd.DataFrame(products)

class RecommendationSystem:
    def __init__(self, df=None):
        if df is None:
            self.df = create_sample_data()
        else:
            self.df = df
        self.tfidf = TfidfVectorizer(stop_words='english')
        
        # Combine relevant features for content-based filtering
        self.df['features'] = self.df.apply(
            lambda x: f"{x['name']} {x['category']} {x['description']}", axis=1
        )
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['features'])
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
    
    def get_recommendations(self, product_id, n_recommendations=5):
        idx = self.df.index[self.df['product_id'] == product_id][0]
        similarity_scores = list(enumerate(self.similarity_matrix[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:n_recommendations+1]
        product_indices = [i[0] for i in similarity_scores]
        return self.df.iloc[product_indices]
    
    def get_product_by_name(self, product_name):
        return self.df[self.df['name'] == product_name].iloc[0]
    
    def get_categories(self):
        return list(self.df['category'].unique())
    
    def get_products_by_category(self, category):
        if category == 'All':
            return self.df
        return self.df[self.df['category'] == category]
    
    def save_model(self, filepath='recommendation_model.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_model(filepath='recommendation_model.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

if __name__ == "__main__":
    # Example usage
    rec_system = RecommendationSystem()
    sample_recommendations = rec_system.get_recommendations(1)
    print("Sample recommendations:", sample_recommendations['name'].tolist())