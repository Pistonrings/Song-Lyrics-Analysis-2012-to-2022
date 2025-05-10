import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from io import BytesIO
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("billboard_2012_to_2022_top_100_song_lyrics.csv") 
    return df
df = load_data()

st.title("🎵 Billboard Top 100 Songs Analysis (2012 - 2022)")
st.write("### Dataset Overview")
st.dataframe(df.head())

st.write("## 🎯 Custom Analysis")
year_range = st.slider("Select Year Range", 2012, 2022, (2015, 2020))
all_genres = df["Genre"].str.split(", ").explode().unique()
selected_genre = st.selectbox("Select Genre", sorted(all_genres))
filtered_df = df[df["Genre"].str.contains(selected_genre, na=False, case=False)]
st.write(f"### Showing analysis for {selected_genre} songs")
st.dataframe(filtered_df[['Top100Year', 'SongTitle', 'Artist']])

# generate insights
st.write("## 📊 Quick Insights")
if not filtered_df.empty:
    avg_lyrics_len = filtered_df['Lyrics'].str.len().mean()
    top_artist = filtered_df['Artist'].value_counts().idxmax()
    st.write(f"🎼 **Average Lyrics Length:** {int(avg_lyrics_len)} characters")
    st.write(f"🎤 **Most Frequent Artist:** {top_artist}")
    # ------ ANALYSIS: Trend of Selected Genre Over Years ------
    st.write(f"## 📈 Trend of '{selected_genre}' Songs Over the Years")
    df_genre_filtered = df[df["Genre"].str.contains(selected_genre, na=False, case=False)]
    genre_trend = df_genre_filtered["Top100Year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(x=genre_trend.index, y=genre_trend.values, marker='o', color='crimson', linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Songs")
    ax.set_title(f"Trend of '{selected_genre}' Songs Over the Years")
    st.pyplot(fig)
else:
    st.write("No data available for the selected criteria.")

# ------ download report ------
st.write("## 📥 Download Analysis Report")
output = BytesIO()
filtered_df.to_csv(output, index=False)
output.seek(0)
st.download_button("Download Filtered Data", output, file_name="filtered_analysis.csv", mime="text/csv")
st.write("### 🚀 Explore trends in Billboard music and discover new insights!")
df["LyricsLength"] = df["Lyrics"].astype(str).apply(len)
df_exploded = df.assign(Genre=df["Genre"].str.split(", ")).explode("Genre")
col1, col2 = st.columns(2)

# --------- ANALYSIS 1: ----------
with col1:
    st.subheader("🎤 Top Artists with Most Billboard Hits")
    top_artists = df['Artist'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(y=top_artists.index, x=top_artists.values, palette="viridis", ax=ax)
    ax.set_xlabel("Number of Songs")
    ax.set_ylabel("Artist")
    ax.set_title("Top 10 Artists with Most Billboard Hits")
    st.pyplot(fig)

# --------- ANALYSIS 2: ----------
with col2:
    st.subheader("🎵 Top 10 Most Popular Genres")
    genre_counts = df["Genre"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='Set2', ax=ax)
    ax.set_xlabel("Genre")
    ax.set_ylabel("Number of Songs")
    ax.set_title("Top 10 Most Popular Genres (2012-2022)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)

# --------- ANALYSIS 3: ----------
with col1:
    st.subheader("🎶 Diversity of Music Genres (2012-2022)")
    genre_diversity = df_exploded.groupby("Top100Year")["Genre"].nunique()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(genre_diversity.index, genre_diversity.values, marker="o", linestyle="-", color="b")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Unique Genres")
    ax.set_title("Diversity of Music Genres (2012-2022)")
    ax.grid(True)
    st.pyplot(fig)

# --------- ANALYSIS 4: ----------
with col2:
    st.subheader("📝 Top 10 Genres by Average Lyrics Length")
    avg_lyrics_by_genre = df_exploded.groupby("Genre")["LyricsLength"].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=avg_lyrics_by_genre.values, y=avg_lyrics_by_genre.index, palette="mako", ax=ax)
    ax.set_xlabel("Average Lyrics Length")
    ax.set_ylabel("Genre")
    ax.set_title("Top 10 Genres with Longest Lyrics")
    st.pyplot(fig)

# --------- ANALYSIS 5:----------
with col1:
    st.subheader("🌟 Most Common Words in Lyrics")
    all_lyrics = " ".join(df['Lyrics'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_lyrics)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# --------- ANALYSIS 6: ----------
with col2:
    st.subheader("📈 Sentiment Analysis of Songs (2012-2022)")
    if "Sentiment" not in df.columns:
        df["Sentiment"] = df["Lyrics"].dropna().apply(lambda x: TextBlob(x).sentiment.polarity)
    sentiment_trend = df.groupby("Top100Year")["Sentiment"].mean()
    sentiment_variance = df.groupby("Top100Year")["Sentiment"].var()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=sentiment_trend.index, y=sentiment_trend.values, marker="o", color="purple", ax=ax)
    ax.set_title("Average Sentiment of Songs (2012-2022)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Sentiment Score")
    ax.grid(True)
    st.pyplot(fig)
    sentiment_trend_change = sentiment_trend.iloc[-1] - sentiment_trend.iloc[0]
    variance_trend_change = sentiment_variance.iloc[-1] - sentiment_variance.iloc[0]
    insights = f"""
    ### **Sentiment Analysis Insights**
    1. **Overall Sentiment Trend:** {"Positive" if sentiment_trend_change > 0 else "Negative"}  
       - Sentiment has **{"increased" if sentiment_trend_change > 0 else "decreased"}** over time.  
    2. **Sentiment Score Change:** {round(sentiment_trend_change, 4)}  
       - Indicates a shift towards more **{"positive" if sentiment_trend_change > 0 else "negative"}** lyrics.  
    3. **Sentiment Variance Trend:** {"Increased" if variance_trend_change > 0 else "Decreased"}  
       - Songs have become **{"more" if variance_trend_change > 0 else "less"}** diverse in emotional tone.  
    4. **Variance Change:** {round(variance_trend_change, 4)}  
       - Suggests lyrics are becoming **{"more emotionally extreme" if variance_trend_change > 0 else "more standardized"}**.  
    """
    st.markdown(insights)

# ------ popularity of Genre Over Years ------
st.write("## 🔥 Genre Popularity Over Years")
df_exploded = df.assign(Genre=df["Genre"].str.split(", ")).explode("Genre")
top_20_genres = df_exploded["Genre"].value_counts().head(20).index
df_filtered = df_exploded[df_exploded["Genre"].isin(top_20_genres)]
genre_year = df_filtered.groupby(["Top100Year", "Genre"]).size().unstack().fillna(0)
fig, ax = plt.subplots(figsize=(12, 6))
genre_year.plot(ax=ax, colormap="tab10", linewidth=2)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Songs")
ax.set_title("Popularity of Top 20 Genres Over the Years")
st.pyplot(fig)
