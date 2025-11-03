import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import io

# -----------------
# APP CONFIGURATION
# -----------------
# This sets the page title, icon, and layout.
st.set_page_config(
    page_title="Spotify Recommender",
    page_icon="🎧",
    layout="centered"
)

# -----------------
# CACHED ML MODEL
# -----------------
# @st.cache_resource tells Streamlit to run this function ONCE and
# store the result. This way, we don't reload the model every time
# the user does something.
@st.cache_resource
def load_model(uploaded_file):
    """Loads data, trains the model, and returns all ML components."""
    try:
        # Read the file from the upload
        data = pd.read_csv(uploaded_file)
        
        # Preferred numeric features (only keep those present)
        preferred = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                     'instrumentalness', 'liveness', 'valence', 'tempo', 'popularity']
        features = [f for f in preferred if f in data.columns]
        
        if not features:
            st.error("Error: No matching numeric feature columns found in dataset.")
            return None, None, None, None

        X = data[features].fillna(0).astype(float)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Build nearest neighbors model
        model = NearestNeighbors(n_neighbors=6, metric='euclidean')
        model.fit(X_scaled)
        
        return data, model, features, scaler
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None, None, None, None

def get_recommendations(song_name, data, model, features, scaler):
    """Finds a song and returns 5 recommendations."""
    if 'track_name' not in data.columns:
        st.error("Error: Dataset does not contain 'track_name' column.")
        return None, None
        
    matches = data[data['track_name'].astype(str).str.contains(song_name, case=False, na=False)]
    if matches.empty:
        st.warning("No matching song found. Try another name.")
        return None, None
    
    # Use the first matched song
    song_idx = matches.index[0]
    orig_track = data.loc[song_idx, 'track_name']
    
    song_vector = data.loc[song_idx, features].fillna(0).astype(float).values.reshape(1, -1)
    song_scaled = scaler.transform(song_vector)
    distances, indices = model.kneighbors(song_scaled)
    
    # Get top 5 recommendations
    rec_idxs = indices[0][1:]
    recommendations = data.loc[rec_idxs].reset_index(drop=True)
    
    # Build display text
    lines = []
    for rank, row in enumerate(recommendations.itertuples(index=False), start=1):
        tn = getattr(row, 'track_name', 'Unknown track')
        artists = getattr(row, 'artists', getattr(row, 'artist', 'Unknown artist'))
        album = getattr(row, 'album_name', '')
        line = f"{rank}. **{tn}** — {artists}" + (f" *({album})*" if album else "")
        lines.append(line)
    
    return orig_track, lines

# -----------------
# UI DESIGN
# -----------------

# Header: This is the green bar from your design
st.markdown(
    """
    <div style="background-color:#1DB954;padding:20px;border-radius:10px;">
        <h1 style="color:white;text-align:center;">Spotify Recommender System</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("## 1. Load Your Dataset")

# File Uploader: This replaces "Load Dataset" button
uploaded_file = st.file_uploader("Upload a CSV file containing song features", type=["csv"])

# This "if" block only runs if a file has been uploaded
if uploaded_file is not None:
    # Load the model (or get it from the cache)
    data, model, features, scaler = load_model(uploaded_file)
    
    # Check if model loading was successful
    if data is not None:
        st.success(f"✅ Loaded dataset with {len(data)} songs!")
        st.markdown("---")
        
        st.markdown("## 2. Get Song Recommendations")
        
        # Text Input: Replaces the popup dialog
        song_name = st.text_input("Enter a song name (case-insensitive):")
        
        # Button: Replaces "Get Song Recommendations" button
        if st.button("Get Recommendations", type="primary"):
            if song_name:
                # Run the logic
                original_song, rec_list = get_recommendations(song_name, data, model, features, scaler)
                
                # Display results
                if original_song and rec_list:
                    st.subheader(f"Top 5 songs similar to '{original_song}':")
                    st.markdown("\n".join(rec_list), unsafe_allow_html=True)
            else:
                st.warning("Please enter a song name.")
