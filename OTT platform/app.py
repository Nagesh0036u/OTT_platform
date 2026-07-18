import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# --- 1. SET YOUR API KEY ---
TMDB_API_KEY = "0fb998f8d54e2dc71d3d391768e2d859"

# --- 2. LOAD & PREP THE DATA ---
# st.cache_data ensures we only download the 22MB dataset once per session!
@st.cache_data
def load_data():
    # We pull a rock-solid, pre-cleaned dataset directly from a GitHub repository
    url = "https://raw.githubusercontent.com/rashida048/Datasets/master/movie_dataset.csv"
    df = pd.read_csv(url)
    
    # We grab all the text features to create a "DNA" profile for each movie
    text_columns = ['genres', 'keywords', 'cast', 'director']
    combined_features = ""
    
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('')
            combined_features += df[col] + " "
            
    df['combined_features'] = combined_features
    return df

df = load_data()

# --- 3. THE MATH (COSINE SIMILARITY) ---
@st.cache_resource
def compute_similarity(data):
    # Convert text into a matrix of token counts, limiting to 5000 words
    cv = CountVectorizer(max_features=5000, stop_words='english')
    count_matrix = cv.fit_transform(data['combined_features'])
    
    # Measure the angle between the vectors (1 = identical, 0 = completely different)
    similarity = cosine_similarity(count_matrix)
    return similarity

similarity = compute_similarity(df)

# --- 4. THE VISUALS (TMDB API) ---
def fetch_poster(movie_title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if 'results' in data and len(data['results']) > 0:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path
                
    except requests.exceptions.ConnectionError:
        pass
    except Exception as e:
        pass
        
    # INSTEAD of a fake placeholder, we return None if a poster is missing
    return None

# --- 5. THE RECOMMENDER LOGIC ---
def recommend(movie):
    try:
        movie_index = df[df.title.str.lower() == movie.lower()].index[0]
    except IndexError:
        return [], []
    
    distances = similarity[movie_index]
    
    # Grab the whole sorted list so we have backup movies if one gets rejected
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:]
    
    recommended_movies = []
    recommended_posters = []
    
    for i in movies_list:
        title = df.iloc[i[0]].title
        poster = fetch_poster(title)
        
        # THE BOUNCER: Only add the movie to the screen IF it has a real poster
        if poster is not None:
            recommended_movies.append(title)
            recommended_posters.append(poster)
            
        # Stop searching the moment we lock in 5 perfect posters
        if len(recommended_movies) == 5:
            break
            
    return recommended_movies, recommended_posters

# --- 6. THE STREAMLIT UI ---
st.set_page_config(page_title="Mini-Netflix", layout="wide")
st.title("🍿 Mini-Netflix Recommender")
st.markdown("Select a movie below, and AI will find 5 titles with similar DNA.")

# Create a sleek dropdown menu
selected_movie = st.selectbox("Pick a movie you love❤️:", df['title'].values)

if st.button("Recommend"):
    if TMDB_API_KEY == "PASTE_YOUR_API_KEY_HERE":
        st.error("Hold up! You forgot to paste your TMDB API key in the code!")
    else:
        with st.spinner("Analyzing movie DNA and fetching posters..."):
            names, posters = recommend(selected_movie)
            
            if names:
                # Create 5 visual columns for the grid layout
                cols = st.columns(5)
                for i in range(5):
                    with cols[i]:
                        st.image(posters[i], use_container_width=True)
                        st.write(f"**{names[i]}**")
            else:
                st.error("Movie not found in the database.")