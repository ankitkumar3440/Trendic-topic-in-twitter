import streamlit as st
from pymongo import MongoClient
from collections import Counter
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import plotly.express as px
from transformers import pipeline
from scipy.cluster.hierarchy import linkage, dendrogram
import networkx as nx
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
nlp = spacy.load("en_core_web_sm")




# Download required NLTK resources
nltk.download('vader_lexicon')

# Set page config
st.set_page_config(page_title="Twitter Analytics Dashboard", layout="wide")

# Initialize models (cached)
@st.cache_resource
def get_emotion_classifier():
    try:
        return pipeline("text-classification", 
                       model="j-hartmann/emotion-english-distilroberta-base", 
                       return_all_scores=False)
    except Exception as e:
        st.error(f"Could not load emotion classifier: {e}")
        return None

@st.cache_resource
def get_spacy_model():
    try:
        return spacy.load("en_core_web_md")
    except OSError:
        st.error("""
            spaCy model 'en_core_web_md' not found. Please install it by running:
            python -m spacy download en_core_web_md
            """)
        return None
    except Exception as e:
        st.error(f"Error loading spaCy model: {e}")
        return None

@st.cache_resource
def get_sentence_transformer():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Could not load sentence transformer: {e}")
        return None

# MongoDB connection with error handling
@st.cache_resource
def init_connection():
    try:
        client = MongoClient(
            "mongodb+srv://kandpaludit0209:sUPFvfQhkQnKV04j@cluster0.vlc3bsc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
            serverSelectionTimeoutMS=5000
        )
        client.server_info()
        return client
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None

client = init_connection()

if client is None:
    st.error("Could not connect to MongoDB. Please check your connection.")
    st.stop()

db = client["twitter_data"]

# Function to get cleaned tweets
@st.cache_data(ttl=300)
def get_cleaned_tweets():
    try:
        collection = db["cleaned_tweets"]
        return [tweet["text"] for tweet in collection.find({}, {"text": 1, "_id": 0}) if "text" in tweet]
    except Exception as e:
        st.error(f"Error fetching cleaned tweets: {e}")
        return []

# Function to get raw tweets
@st.cache_data(ttl=300)
def get_raw_tweets():
    try:
        collection = db["tweets"]
        tweets = [tweet["text"] for tweet in collection.find({}, {"text": 1, "_id": 0}) if "text" in tweet]
        return tweets
    except Exception as e:
        st.error(f"Error fetching raw tweets: {e}")
        return []

# Text cleaning function
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    return text.strip()

# Sentiment Analysis Functions
@st.cache_data(ttl=300)
def analyze_sentiment(tweets):
    sid = SentimentIntensityAnalyzer()
    
    df = pd.DataFrame(tweets, columns=["text"])
    df['clean_text'] = df['text'].apply(clean_text)
    
    # TextBlob analysis
    df['textblob_polarity'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['textblob_sentiment'] = df['textblob_polarity'].apply(
        lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral'))
    
    # VADER analysis
    df['vader_scores'] = df['clean_text'].apply(lambda x: sid.polarity_scores(x))
    df['vader_compound'] = df['vader_scores'].apply(lambda x: x['compound'])
    df['vader_sentiment'] = df['vader_compound'].apply(
        lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))
    
    # Final sentiment (VADER primary)
    df['sentiment'] = df['vader_sentiment']
    
    # Extract hashtags
    df['hashtags'] = df['text'].apply(lambda t: re.findall(r"#(\w+)", t.lower()))
    
    return df

# Emotion Analysis Function
@st.cache_data
def analyze_emotion(_emotion_classifier, tweets):
    df = pd.DataFrame(tweets, columns=["text"])
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Apply emotion classification in batches to avoid memory issues
    emotions = []
    for text in df['clean_text']:
        try:
            result = _emotion_classifier(text[:512])  # Limit to 512 tokens
            emotions.append(result[0]['label'])
        except Exception as e:
            st.warning(f"Error processing text: {e}")
            emotions.append('unknown')
    
    df['emotion'] = emotions
    df['hashtags'] = df['text'].apply(lambda t: re.findall(r"#(\w+)", t.lower()))
    
    return df

# Clustering and Recommendation Functions (Fixed with underscores for unhashable params)
@st.cache_data(ttl=300)
def prepare_clusters(_sbert_model, tweets, _nlp_model, n_clusters=5):
    try:
        # Generate embeddings
        sentence_vectors = _sbert_model.encode(tweets, show_progress_bar=False)
        
        # Dimensionality reduction
        pca = PCA(n_components=50)
        reduced_vectors = pca.fit_transform(sentence_vectors)
        
        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(reduced_vectors)
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': tweets,
            'clean_text': [clean_text(t) for t in tweets],
            'cluster': clusters
        })
        
        # Extract keywords
        def extract_keywords(text):
            doc = _nlp_model(text)
            return [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop]
        
        df['keywords'] = df['clean_text'].apply(extract_keywords)
        
        # Determine cluster topics
        cluster_topics = {}
        for cluster_id in range(n_clusters):
            cluster_keywords = df[df['cluster'] == cluster_id]['keywords'].sum()
            most_common = Counter(cluster_keywords).most_common(3)
            cluster_topics[cluster_id] = ", ".join([word for word, _ in most_common])
        
        df['cluster_topic'] = df['cluster'].map(cluster_topics)
        
        return df, sentence_vectors, pca, kmeans, cluster_topics
    except Exception as e:
        st.error(f"Error in clustering: {e}")
        return None, None, None, None, None

def recommend_by_cluster(query_text, _sbert_model, pca, kmeans, df, sentence_vectors, cluster_topics, top_n=5):
    try:
        query_vec = _sbert_model.encode([query_text])
        query_cluster = kmeans.predict(pca.transform(query_vec))[0]
        cluster_df = df[df['cluster'] == query_cluster]
        cluster_vectors = sentence_vectors[cluster_df.index]
        scores = cosine_similarity(query_vec, cluster_vectors).flatten()
        top_idx = scores.argsort()[-top_n:][::-1]
        
        result_df = cluster_df.iloc[top_idx][['text', 'cluster_topic']].copy()
        result_df['similarity_score'] = scores[top_idx]
        result_df = result_df.reset_index(drop=True)
        
        return result_df, cluster_topics[query_cluster]
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return pd.DataFrame(), ""

# App title
st.title("Twitter Analytics Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
analysis_type = st.sidebar.radio(
    "Select Analysis Type",
    ["Trending Topics", "Hashtag Analysis", "Sentiment Analysis", 
     "Emotion Detection", "Word Clustering", "Content Recommendations"]
)

# Debug controls
st.sidebar.markdown("---")
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("Cache cleared!")



elif analysis_type == "Trending Topics":
    st.header("Trending Topics Analysis")
    cleaned_tweets = get_cleaned_tweets()
    
    if cleaned_tweets:
        word_counts = Counter(" ".join(cleaned_tweets).split())
        trending_words = word_counts.most_common(10)
        df = pd.DataFrame(trending_words, columns=["Word", "Frequency"])
        
        st.dataframe(df)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df, x="Word", y="Frequency", palette="viridis", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No cleaned tweets found in the database.")

elif analysis_type == "Hashtag Analysis":
    st.header("Hashtag Analysis")
    tweets = get_raw_tweets()
    
    if tweets:
        hashtags = [word.lower() for tweet in tweets for word in tweet.split() if word.startswith("#")]
        
        if hashtags:
            top_hashtags = Counter(hashtags).most_common(10)
            df = pd.DataFrame(top_hashtags, columns=["Hashtag", "Frequency"])
            
            st.dataframe(df)
            
            fig = px.bar(df, x="Hashtag", y="Frequency", title="Top Hashtags")
            st.plotly_chart(fig)
        else:
            st.warning("No hashtags found in tweets")
    else:
        st.warning("No tweets found in the database.")

elif analysis_type == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    tweets = get_raw_tweets()
    
    if tweets:
        sentiment_df = analyze_sentiment(tweets)
        
        # Overall sentiment
        st.subheader("Overall Sentiment Distribution")
        sentiment_counts = sentiment_df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        fig = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                    title='Overall Sentiment Analysis',
                    color='Sentiment',
                    color_discrete_map={'positive':'green', 'neutral':'gray', 'negative':'red'})
        st.plotly_chart(fig)
        
        # Sentiment by hashtag
        st.subheader("Sentiment Analysis by Hashtag")
        all_hashtags = list(set([h for sublist in sentiment_df['hashtags'] for h in sublist]))
        
        if all_hashtags:
            selected_hashtag = st.selectbox("Select a hashtag", all_hashtags)
            
            if selected_hashtag:
                hashtag_tweets = sentiment_df[sentiment_df['hashtags'].apply(lambda x: selected_hashtag in x)]
                
                if not hashtag_tweets.empty:
                    hashtag_counts = hashtag_tweets['sentiment'].value_counts().reset_index()
                    hashtag_counts.columns = ['Sentiment', 'Count']
                    
                    fig = px.pie(hashtag_counts, values='Count', names='Sentiment',
                                title=f'Sentiment for #{selected_hashtag}',
                                color='Sentiment',
                                color_discrete_map={'positive':'green', 'neutral':'gray', 'negative':'red'})
                    st.plotly_chart(fig)
                    
                    st.dataframe(hashtag_tweets[['text', 'sentiment']].head(10))
                else:
                    st.warning(f"No tweets found with hashtag #{selected_hashtag}")
        else:
            st.warning("No hashtags found in tweets")
    else:
        st.warning("No tweets found in the database.")

elif analysis_type == "Emotion Detection":
    st.header("Emotion Detection")
    
    with st.spinner("Loading emotion classifier (this may take a minute)..."):
        emotion_classifier = get_emotion_classifier()
    
    if emotion_classifier:
        tweets = get_raw_tweets()
        
        if tweets:
            with st.spinner("Analyzing emotions in tweets..."):
                emotion_df = analyze_emotion(emotion_classifier, tweets)
            
            # Overall emotion distribution
            st.subheader("Overall Emotion Distribution")
            emotion_counts = emotion_df['emotion'].value_counts().reset_index()
            emotion_counts.columns = ['Emotion', 'Count']
            
            # Define a color map for emotions
            emotion_colors = {
                'anger': 'red',
                'disgust': 'darkgreen',
                'fear': 'purple',
                'joy': 'gold',
                'neutral': 'gray',
                'sadness': 'blue',
                'surprise': 'orange',
                'unknown': 'black'
            }
            
            fig = px.pie(emotion_counts, values='Count', names='Emotion',
                        title='Emotion Distribution in Tweets',
                        color='Emotion',
                        color_discrete_map=emotion_colors)
            st.plotly_chart(fig)
            
            # Emotion by hashtag
            st.subheader("Emotion Analysis by Hashtag")
            all_hashtags = list(set([h for sublist in emotion_df['hashtags'] for h in sublist]))
            
            if all_hashtags:
                selected_hashtag = st.selectbox("Select a hashtag to analyze emotions", all_hashtags)
                
                if selected_hashtag:
                    hashtag_tweets = emotion_df[emotion_df['hashtags'].apply(lambda x: selected_hashtag in x)]
                    
                    if not hashtag_tweets.empty:
                        hashtag_emotions = hashtag_tweets['emotion'].value_counts().reset_index()
                        hashtag_emotions.columns = ['Emotion', 'Count']
                        
                        fig = px.pie(hashtag_emotions, values='Count', names='Emotion',
                                    title=f'Emotions for #{selected_hashtag}',
                                    color='Emotion',
                                    color_discrete_map=emotion_colors)
                        st.plotly_chart(fig)
                        
                        st.subheader(f"Sample Tweets for #{selected_hashtag}")
                        st.dataframe(hashtag_tweets[['text', 'emotion']].head(10))
                    else:
                        st.warning(f"No tweets found with hashtag #{selected_hashtag}")
            else:
                st.warning("No hashtags found in tweets")
        else:
            st.warning("No tweets found in the database.")

elif analysis_type == "Word Clustering":
    st.header("Word Clustering and Network Analysis")
    tweets = get_cleaned_tweets()
    
    if tweets:
        st.subheader("Word Similarity Network")
        
        # Create co-occurrence matrix
        words = [word for tweet in tweets for word in tweet.split()]
        word_counts = Counter(words)
        top_words = [word for word, count in word_counts.most_common(30)]  # Top 30 words
        
        # Create Jaccard similarity matrix
        word_presence = {}
        for word in top_words:
            word_presence[word] = [1 if word in tweet.split() else 0 for tweet in tweets]
        
        jaccard_df = pd.DataFrame(word_presence)
        
        # Create graph
        G = nx.Graph()
        threshold = 0.1  # Similarity threshold
        
        for i in range(len(top_words)):
            for j in range(i+1, len(top_words)):
                word1 = top_words[i]
                word2 = top_words[j]
                similarity = sum((jaccard_df[word1] & jaccard_df[word2])) / sum((jaccard_df[word1] | jaccard_df[word2]))
                if similarity > threshold:
                    G.add_edge(word1, word2, weight=similarity)
        
        # Draw network
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5)
        nx.draw_networkx_nodes(G, pos, node_size=500, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        plt.title("Word Similarity Network")
        st.pyplot(fig)
        
        # Show dendrogram
        st.subheader("Hierarchical Clustering of Words")
        linked = linkage(jaccard_df.T, method='ward')
        fig, ax = plt.subplots(figsize=(12, 6))
        dendrogram(linked, labels=top_words, leaf_rotation=45, ax=ax)
        plt.title("Word Clustering Dendrogram")
        st.pyplot(fig)
    else:
        st.warning("No cleaned tweets found in the database.")

elif analysis_type == "Content Recommendations":
    st.header("Content-Based Tweet Recommendations")
    
    with st.spinner("Loading NLP models (this may take a minute)..."):
        sbert = get_sentence_transformer()
        nlp = get_spacy_model()
    
    if sbert and nlp:
        tweets = get_raw_tweets()
        
        if tweets:
            with st.spinner("Preparing tweet clusters..."):
                cluster_df, sentence_vectors, pca, kmeans, cluster_topics = prepare_clusters(
                    _sbert_model=sbert,
                    tweets=tweets,
                    _nlp_model=nlp
                )
            
            if cluster_df is not None:
                st.subheader("Tweet Clusters Overview")
                
                # Display cluster information
                cluster_info = []
                for cluster_id, group in cluster_df.groupby('cluster'):
                    cluster_info.append({
                        "Cluster": cluster_id + 1,
                        "Topic": cluster_topics[cluster_id],
                        "Number of Tweets": len(group)
                    })
                
                st.dataframe(pd.DataFrame(cluster_info))
                
                # Recommendation system
                st.subheader("Find Similar Tweets")
                user_query = st.text_input("Enter keywords or a tweet to find similar content:")
                
                if user_query:
                    with st.spinner("Finding similar tweets..."):
                        recommendations, matched_topic = recommend_by_cluster(
                            query_text=user_query,
                            _sbert_model=sbert,
                            pca=pca,
                            kmeans=kmeans,
                            df=cluster_df,
                            sentence_vectors=sentence_vectors,
                            cluster_topics=cluster_topics
                        )
                    
                    st.write(f"**Matched Cluster Topic:** {matched_topic}")
                    st.dataframe(recommendations[['text', 'similarity_score']])
                
                # Cluster visualization
                st.subheader("Cluster Visualization")
                reduced_vectors_2d = PCA(n_components=2).fit_transform(pca.transform(sentence_vectors))
                
                fig = px.scatter(
                    x=reduced_vectors_2d[:, 0],
                    y=reduced_vectors_2d[:, 1],
                    color=cluster_df['cluster'].astype(str),
                    hover_data={'text': cluster_df['text']},
                    labels={'color': 'Cluster'},
                    title="Tweet Clusters (2D Projection)"
                )
                st.plotly_chart(fig)
            else:
                st.error("Failed to create tweet clusters")
        else:
            st.warning("No tweets found in the database.")
    else:
        st.warning("Required NLP models could not be loaded.")

# Add info to sidebar
st.sidebar.markdown("---")
st.sidebar.info("Twitter Analytics Dashboard v2.0")
