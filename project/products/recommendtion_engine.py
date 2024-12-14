from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.db.models import Count, Sum


from .models import UserInteraction ,Product

# ml_recommender.py
import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics.pairwise import cosine_similarity
# from django.core.cache import cache

# Deep Learning Recommender
import torch
import torch.nn as nn
import torch.optim as optim

from products.bulk_adder import add_products

# add_products(1)

class DeepRecommenderModel(nn.Module):
    """
    Neural Network-based Recommendation Model
    """
    def __init__(self, num_users, num_products, embedding_dim=50):
        super(DeepRecommenderModel, self).__init__()

        # User and Product Embedding Layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.product_embedding = nn.Embedding(num_products, embedding_dim)

        # Neural network layers
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

        # Weight initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, user, product):
        # Embed users and products
        user_embed = self.user_embedding(user)
        product_embed = self.product_embedding(product)

        # Concatenate embeddings
        combined = torch.cat([user_embed, product_embed], dim=1)

        # Pass through neural network
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)

        # Output interaction probability
        output = torch.sigmoid(self.output(x))
        return output

class MLRecommendationEngine:
    def __init__(self):
        self.model = None
        self.user_mapping = {}
        self.product_mapping = {}
        self.inverse_user_mapping = {}
        self.inverse_product_mapping = {}
        
        # Enhanced tracking of product interactions
        self.product_interaction_count = {}
        self.product_popularity = {}

    def prepare_training_data(self, interactions):
        """
        Enhanced method to capture comprehensive product information
        """
        # Extract unique users and products
        unique_users = interactions['user_id'].unique()
        unique_products = interactions['product_id'].unique()
        
        # Create mappings
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.product_mapping = {product: idx for idx, product in enumerate(unique_products)}
        
        self.inverse_user_mapping = {v: k for k, v in self.user_mapping.items()}
        self.inverse_product_mapping = {v: k for k, v in self.product_mapping.items()}
        
        # Count product interactions
        self.product_interaction_count = interactions['product_id'].value_counts().to_dict()
        
        # Calculate product popularity with weighted approach
        self.product_popularity = interactions.groupby('product_id')['weight'].agg(['sum', 'count'])
        
        # Transform original user and product IDs to numerical indices
        interactions['user_idx'] = interactions['user_id'].map(self.user_mapping)
        interactions['product_idx'] = interactions['product_id'].map(self.product_mapping)
        
        return interactions

    def get_recommendations(self, user_id, top_n=5):
        """
        Advanced recommendation method with comprehensive product inclusion strategy
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_recommendation_model first.")
        
        # Check if user exists in training data
        user_idx = self.user_mapping.get(user_id)
        
        # Prepare full product tensor
        all_product_indices = list(range(len(self.product_mapping)))
        
        if user_idx is not None:
            # Personalized recommendations for existing users
            user_tensor = torch.LongTensor([user_idx] * len(self.product_mapping))
            product_tensor = torch.LongTensor(all_product_indices)
            
            with torch.no_grad():
                predictions = self.model(user_tensor, product_tensor)
            
            # Sort predictions
            sorted_indices = predictions.squeeze().argsort(descending=True)
            
            # Recommendation selection strategy
            recommendations = []
            seen_products = set()
            
            for idx in sorted_indices:
                product_id = self.inverse_product_mapping[idx.item()]
                
                # Ensure diversity and include less-seen products
                if product_id not in seen_products:
                    recommendations.append(product_id)
                    seen_products.add(product_id)
                
                if len(recommendations) >= top_n:
                    break
        
        else:
            # Fallback for new users: Sophisticated recommendation strategy
            # Combine popularity and low-interaction products
            recommendations = self._generate_diverse_recommendations(top_n)
        
        return recommendations

    def _generate_diverse_recommendations(self, top_n):
        """
        Generate recommendations for new users with product diversity
        """
        # Sort products by a combination of popularity and interaction count
        # This ensures both popular and less-seen products get a chance
        product_scores = {}
        for product, data in self.product_popularity.iterrows():
            # Complex scoring mechanism
            popularity_score = data['sum']  # Total interaction weight
            interaction_count = data['count']  # Number of interactions
            novelty_score = 1 / (interaction_count + 1)  # Bonus for less-seen products
            
            # Weighted combination of scores
            product_scores[product] = (
                0.6 * popularity_score + 
                0.4 * (novelty_score * len(self.product_mapping))
            )
        
        # Sort products by the combined score
        sorted_products = sorted(
            product_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Select top N diverse products
        recommendations = [product for product, _ in sorted_products[:top_n]]
        
        return recommendations

    def train_recommendation_model(self, interactions):
        """
        Modified training method to ensure all products are considered
        """
        # Prepare data with numerical indices and capture product information
        df = self.prepare_training_data(interactions)
        
        # Convert data to PyTorch tensors
        users = torch.LongTensor(df['user_idx'].values)
        products = torch.LongTensor(df['product_idx'].values)
        
        # Ensure targets are float tensor with correct shape
        targets = torch.FloatTensor(df['weight'].values).unsqueeze(1)
        
        # Initialize recommendation model
        self.model = DeepRecommenderModel(
            num_users=len(self.user_mapping),
            num_products=len(self.product_mapping)
        )
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(50):
            # Forward pass
            predictions = self.model(users, products)
            
            # Compute loss
            loss = criterion(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f'Epoch [{epoch+1}/50], Loss: {loss.item():.4f}')

# Example usage remains the same as in previous example

# class MLRecommendationEngine:
#     def __init__(self):
#         # Existing initializations
#         self.model = None
#         self.user_mapping = {}
#         self.product_mapping = {}
#         self.inverse_user_mapping = {}
#         self.inverse_product_mapping = {}
        
#         # New: Store global popularity information
#         self.product_popularity = {}
    
#     def prepare_training_data(self, interactions):
#         """
#         Enhanced method to capture product popularity
#         """
#         # Existing mapping logic
#         unique_users = interactions['user_id'].unique()
#         unique_products = interactions['product_id'].unique()
        
#         self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
#         self.product_mapping = {product: idx for idx, product in enumerate(unique_products)}
        
#         self.inverse_user_mapping = {v: k for k, v in self.user_mapping.items()}
#         self.inverse_product_mapping = {v: k for k, v in self.product_mapping.items()}
        
#         # Calculate product popularity
#         product_popularity = interactions.groupby('product_id')['weight'].sum()
#         self.product_popularity = product_popularity.sort_values(ascending=False)
        
#         # Transform original user and product IDs to numerical indices
#         interactions['user_idx'] = interactions['user_id'].map(self.user_mapping)
#         interactions['product_idx'] = interactions['product_id'].map(self.product_mapping)
        
#         return interactions
    

#     def train_recommendation_model(self, interactions):
#         """
#         Train deep learning recommendation model
        
#         Args:
#             interactions (pd.DataFrame): User-product interaction data
#         """
#         # Prepare data with numerical indices
#         df = self.prepare_training_data(interactions)
        
#         # Convert data to PyTorch tensors for model training
#         users = torch.LongTensor(df['user_idx'].values)      # User indices
#         products = torch.LongTensor(df['product_idx'].values)  # Product indices
        
#         # Ensure targets are float tensor with correct shape
#         targets = torch.FloatTensor(df['weight'].values).unsqueeze(1)
        
#         # Initialize recommendation model
#         self.model = DeepRecommenderModel(
#             num_users=len(self.user_mapping),     # Total unique users
#             num_products=len(self.product_mapping)  # Total unique products
#         )   
    
#         # Define loss function (Binary Cross-Entropy)
#         # Measures difference between predicted and actual interaction probabilities
#         criterion = nn.BCELoss()
        
#         # Define optimizer (Adam) for updating model weights
#         # Adaptive learning rate helps faster and more efficient convergence
#         optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
#         # Training loop to improve model prediction
#         for epoch in range(50):  # 50 training iterations
#             # Forward pass: Generate predictions
#             predictions = self.model(users, products)
            
#             # Compute loss between predictions and actual interactions
#             loss = criterion(predictions, targets)
            
#             # Backward pass: Compute gradients
#             optimizer.zero_grad()   # Reset previous gradients
#             loss.backward()         # Compute gradient of loss
#             optimizer.step()        # Update model weights
            
#             # Print training progress
#             print(f'Epoch [{epoch+1}/50], Loss: {loss.item():.4f}')

#     def get_recommendations(self, user_id, top_n=5):
#         """
#         Enhanced recommendation method with fallback strategy
#         """
#         # Check if model is trained
#         if self.model is None:
#             raise ValueError("Model not trained. Call train_recommendation_model first.")
        
#         # Check if user exists in training data
#         user_idx = self.user_mapping.get(user_id)
        
#         if user_idx is not None:
#             # Existing user: Use personalized recommendations
#             user_tensor = torch.LongTensor([user_idx] * len(self.product_mapping))
#             product_tensor = torch.LongTensor(list(range(len(self.product_mapping))))
            
#             with torch.no_grad():
#                 predictions = self.model(user_tensor, product_tensor)
            
#             top_product_indices = predictions.squeeze().argsort(descending=True)[:top_n]
            
#             recommendations = [
#                 self.inverse_product_mapping[idx.item()] 
#                 for idx in top_product_indices
#             ]
#         else:
#             # Fallback for new users: Recommend most popular products
#             recommendations = list(self.product_popularity.head(top_n).index)
        
#         return recommendations

# # Example usage demonstrating different scenarios
# def generate_recommendations_for_users(recommendation_engine, user_ids, top_n=5):
#     """
#     Generate recommendations for multiple users, handling both existing and new users
#     """
#     recommendations = {}
#     for user_id in user_ids:
#         try:
#             user_recommendations = recommendation_engine.get_recommendations(user_id, top_n)
#             recommendations[user_id] = user_recommendations
#         except ValueError as e:
#             print(f"Error for user {user_id}: {e}")
    
#     return recommendations

# # Practical implementation example
# def recommendation_workflow():
#     # 1. Prepare interaction data
#     interactions_df = pd.DataFrame([
#         {'user_id': 1, 'product_id': 101, 'weight': 0.9},
#         {'user_id': 1, 'product_id': 102, 'weight': 0.7},
#         {'user_id': 2, 'product_id': 101, 'weight': 0.6},
#         # More interaction records...
#     ])
    
#     # 2. Initialize and train recommendation engine
#     recommendation_engine = MLRecommendationEngine()
#     recommendation_engine.train_recommendation_model(interactions_df)
    
#     # 3. Generate recommendations for various users
#     # Mix of existing and new users
#     user_ids_to_recommend = [1, 2, 3, 4, 999]
#     recommendations = generate_recommendations_for_users(
#         recommendation_engine, 
#         user_ids_to_recommend
#     )
    
#     # Print recommendations
#     for user_id, user_recommendations in recommendations.items():
#         print(f"Recommendations for User {user_id}: {user_recommendations}")

# # Run the workflow
# recommendation_workflow()



# other engine without machine learning
# class EcommerceRecommendationEngine:
#     def __init__(self):
#         self.content_vectorizer = TfidfVectorizer(stop_words='english')
#         self.product_matrix = None

#     def prepare_product_features(self, products):
#         """
#         Prepare product features for content-based recommendations
#         """
#         # Combine text features for each product
#         product_features = [
#             f"{product.name} {product.description} {product.category} {product.brand}"
#             for product in products
#         ]

#         # Convert to TF-IDF matrix
#         self.product_matrix = self.content_vectorizer.fit_transform(product_features)
#         return self.product_matrix

#     def content_based_recommendations(self, product, products, top_n=5):
#         """
#         Generate content-based recommendations
#         """
#         if self.product_matrix is None:
#             raise ValueError("Product matrix not prepared. Call prepare_product_features first.")

#         # Get TF-IDF vector for target product
#         target_vector = self.content_vectorizer.transform([
#             f"{product.name} {product.description} {product.category} {product.brand}"
#         ])

#         # Calculate cosine similarity
#         similarities = cosine_similarity(target_vector, self.product_matrix)[0]

#         # Sort and get top recommendations (excluding the input product)
#         similar_indices = similarities.argsort()[::-1]
#         recommendations = [
#             products[idx] for idx in similar_indices
#             if products[idx].id != product.id
#         ][:top_n]

#         return recommendations

#     def collaborative_filtering(self, user, products, top_n=5):
#         """
#         Generate collaborative filtering recommendations
#         """
#         # Get user's past interactions
#         interactions = UserInteraction.objects.filter(user=user)

#         # Create user-product interaction matrix
#         interaction_data = interactions.values(
#             'user_id', 'product_id', 'weight', 'interaction_type'
#         )

#         # Convert to DataFrame
#         df = pd.DataFrame(list(interaction_data))

#         # Weight interactions differently
#         interaction_weights = {
#             'view': 1.0,
#             'cart': 2.0,
#             'wishlist': 1.5,
#             'purchase': 3.0,
#             'review': 2.5
#         }
#         df['weighted_interaction'] = df['interaction_type'].map(interaction_weights) * df['weight']

#         # Aggregate user preferences
#         user_preferences = df.groupby('product_id')['weighted_interaction'].sum()

#         # Sort and get top recommendations
#         top_recommendations = user_preferences.nlargest(top_n)

#         recommended_products = [
#             Product.objects.get(id=product_id)
#             for product_id in top_recommendations.index
#         ]

#         return recommended_products

#     def trending_recommendations(self, products, time_window=30, top_n=5):
#         """
#         Generate trending product recommendations
#         """
#         # Get interactions from last 30 days
#         recent_interactions = UserInteraction.objects.filter(
#             timestamp__gte=timezone.now() - timezone.timedelta(
#                 days=time_window)
#         )

#         # Count and rank trending products
#         trending_products = recent_interactions.values('product_id').annotate(
#             interaction_count=Count('id'),
#             total_weight=Sum('weight')
#         ).order_by('-interaction_count', '-total_weight')

#         # Get top trending products
#         recommended_products = [
#             Product.objects.get(id=item['product_id'])
#             for item in trending_products[:top_n]
#         ]

#         return recommended_products
