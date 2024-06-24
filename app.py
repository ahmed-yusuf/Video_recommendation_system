import pandas as pd
import numpy as np

import json

#Loading and inspecting the JSON content
with open('data.json') as f:
    data = json.load(f)

# Printing the loaded JSON content to understand its structure
print(json.dumps(data, indent=2))

data.keys()

tst = pd.json_normalize(data['users'])

len(tst.watch_history[3])

df = pd.json_normalize(data, record_path=['users', 'watch_history'], 
                       meta=[['users', 'user_id'], ['users', 'name']])

df



# Convert 'users' data to DataFrame
users_df = pd.json_normalize(data['users'])
print(users_df)

# Convert 'videos' data to DataFrame
videos_df = pd.json_normalize(data['videos'])
print(videos_df)

users_df

len(users_df['watch_history'])

videos_df.head(2)

#importing necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity




def create_user_profile(user_id):
    watch_history = users_df[users_df['user_id'] == user_id]['watch_history'].iloc[0]
    watched_videos = videos_df[videos_df['video_id'].isin(watch_history)]
    return ' '.join(watched_videos['tags'].sum())



users_df['profile'] = users_df['user_id'].apply(create_user_profile)

# Feature extraction
tfidf = TfidfVectorizer()
user_profiles = tfidf.fit_transform(users_df['profile'])
video_features = tfidf.transform(videos_df['tags'].apply(lambda x: ' '.join(x)))

# Similarity calculation
def get_recommendations(user_id, top_n=5):
    user_profile = user_profiles[users_df['user_id'] == user_id]
    similarities = cosine_similarity(user_profile, video_features)
    
    # Get top N recommendations
    top_indices = similarities.argsort()[0][-top_n:][::-1]
    return videos_df.iloc[top_indices]