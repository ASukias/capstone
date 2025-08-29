import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import gradio as gr

load_dotenv()

# === Load and prepare data ===
books = pd.read_csv('books_with_emotions.csv')
books['isbn13'] = books['isbn13'].astype(str)  # Ensure ISBNs are strings
books['large_thumbnail'] = books['thumbnail'] + "&fife=w800"
books['large_thumbnail'] = np.where(
    books['large_thumbnail'].isna(),
    "cover-not-found.jpg",
    books['large_thumbnail'],
)

# === Embed documents ===
raw_documents = TextLoader('tagged_description.txt').load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

# === Helper for placeholder plot ===
def create_placeholder_plot(message):
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=14)
    ax.axis('off')
    return fig

# === Recommendation logic ===
def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    # Extract ISBNs as strings
    books_list = [rec.page_content.strip('"').split()[0] for rec in recs]
    books['isbn13'] = books['isbn13'].astype(str)
    book_recs = books[books['isbn13'].isin(books_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs['simple_categories'] == category].head(final_top_k)

    # Use the correct emotion columns as per your CSV
    if tone == "Happy":
        book_recs = book_recs.sort_values(by='joy', ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by='surprise', ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by='anger', ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by='fear', ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by='sadness', ascending=False)

    return book_recs

def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)

    # Handle empty recommendations
    if recommendations.empty:
        placeholder_fig = create_placeholder_plot("No recommendations found.")
        return [], placeholder_fig, placeholder_fig, placeholder_fig

    results = []
    for _, row in recommendations.iterrows():
        description = row['description']
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."
        authors_split = row['authors'].split(';')
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row['large_thumbnail'], caption))

    # === Visualization 1: Rating distribution ===
    fig1, ax1 = plt.subplots()
    # Defensive check for column existence
    if 'average_rating' in recommendations.columns and not recommendations['average_rating'].isnull().all():
        sns.histplot(data=recommendations, x='average_rating', bins=10, kde=True, ax=ax1, color='skyblue')
        ax1.set_title('Average Rating Distribution of Recommendations')
        ax1.set_xlabel('Average Rating')
        ax1.set_ylabel('Count')
    else:
        ax1.text(0.5, 0.5, "No rating data", ha='center', va='center', fontsize=14)
        ax1.axis('off')
    plt.tight_layout()

    # === Visualization 2 & 3: PCA & Similarity Matrix ===
    all_vectors = db_books._collection.get(include=["embeddings"])["embeddings"]
    isbn_to_vector = dict(zip(books['isbn13'], all_vectors))
    vectors_for_recs = [isbn_to_vector[isbn] for isbn in recommendations['isbn13'] if isbn in isbn_to_vector]

    if len(vectors_for_recs) < 2:
        fig2 = create_placeholder_plot("Not enough data for PCA plot.")
        fig3 = create_placeholder_plot("Not enough data for similarity matrix.")
    else:
        # PCA plot
        pca = PCA(n_components=2)
        coords = pca.fit_transform(vectors_for_recs)
        fig2, ax2 = plt.subplots()
        scatter = ax2.scatter(coords[:, 0], coords[:, 1], c=recommendations['joy'], cmap='coolwarm', s=50)
        ax2.set_title("Book Clusters (PCA, colored by Joy)")
        ax2.set_xlabel("Component 1")
        ax2.set_ylabel("Component 2")
        plt.colorbar(scatter, ax=ax2)

        # Similarity matrix plot
        sim_matrix = cosine_similarity(vectors_for_recs)
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        sns.heatmap(sim_matrix, cmap="viridis", ax=ax3)
        ax3.set_title("Similarity Matrix of Recommended Books")
        plt.tight_layout()

    return results, fig1, fig2, fig3

# === Gradio UI ===
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Base()) as dashboard:
    gr.Markdown("## 📚 Semantic Book Recommender")
    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:", placeholder="A story about nature and curiosity")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find Recommendations")
    gr.Markdown("## AI-Recommended Books")
    output_gallery = gr.Gallery(label="Books", columns=4, rows=2, height=400)
    gr.Markdown("## Data Visualizations")
    output_plot1 = gr.Plot(label="Rating Distribution")
    output_plot2 = gr.Plot(label="PCA Book Clusters")
    output_plot3 = gr.Plot(label="Book Similarity Matrix")

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=[output_gallery, output_plot1, output_plot2, output_plot3]
    )

if __name__ == '__main__':
    dashboard.launch()
