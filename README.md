# Python AI-Powered Book Recommendation Platform

## Overview

This project is a web-based AI platform designed to counteract the declining reading proficiency and engagement among students in the United States. Leveraging advanced **Natural Language Processing (NLP)** and **Machine Learning (ML)**, the system provides personalized book recommendations by analyzing over 7,000 books, considering metadata, emotional tone, and semantic content.

The platform aims to help users discover books that match their interests and current moods, fostering a culture of reading and improving literacy outcomes.

---

## Features

- **Personalized Recommendations:** Users input descriptive queries and select optional categories and emotional tones to receive curated book suggestions.
- **Emotion Analysis:** Utilizes transformer-based classification models to quantify emotional undertones in book descriptions.
- **Semantic Search:** Employs OpenAI embeddings combined with a vector database (ChromaDB) for fast, relevant book retrieval based on semantic similarity.
- **Interactive Dashboard:** Built with **Gradio**, featuring visualization components such as rating histograms, PCA-based clustering, and similarity matrices.
- **Cloud-Native Deployment:** Designed for scalable cloud deployment using containerized services with Docker and AWS.
- **Ethical and Transparent:** Implements bias mitigation strategies and ensures transparency in recommendations.

---

## Technologies Used

- Programming Languages: **Python**, **Java**, **SQL**, **JavaScript (Angular)**
- Frameworks & Libraries: **Spring Boot**, **Gradio**, **Angular**, **Hugging Face Transformers**, **scikit-learn**, **Pandas**, **NumPy**
- Databases: **MySQL**, **ChromaDB (vector store)**
- Machine Learning Services: **OpenAI API (embedding models)**, **Transformer-based emotion and category classifiers**
- DevOps: **Docker**, **AWS Cloud Services**, **CI/CD pipelines**
- Version Control: **Git**, **GitLab**

---

## Installation & Setup

### Prerequisites

- Python 3.9 or higher (Python 3.12 recommended)
- Java JDK (for backend)
- Node.js & npm (for Angular frontend)
- Docker
- Git

### Installation Steps

1. Clone the repository:
git clone <repo-url>
cd <project-folder>

2. Create and activate a Python virtual environment:
python -m venv venv

Windows
venv\Scripts\activate

macOS/Linux
source venv/bin/activate

3. Install Python dependencies:
kagglehub
pandas
matplotlib
seaborn
python-dotenv
langchain-community
langchain-opencv
langchain-chroma
transformers
gradio
notebook
ipywidgets

4. Acquire API keys:
- OpenAI API key
- Hugging Face API token

5. Create a `.env` file in the project root with:
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_API_TOKEN=your_huggingface_api_token

6. Run backend (Spring Boot) and frontend (Angular) according to the project folder structure instructions.

7. To run the recommendation dashboard:
   python gradio-dashboard.py


8. Visit the provided local URL to interact with the application.

---

## Usage

1. Enter a description of the kind of book you are interested in.
2. Optionally select a category and desired emotional tone.
3. Submit the query to receive personalized recommendations.
4. View interactive visualizations showing rating distributions, semantic clusters, and similarity scores.
5. Refine your inputs to explore different recommendations.

---

## Project Structure

- `backend/`: Java Spring Boot REST API services.
- `frontend/`: Angular web application.
- `dashboard/`: Python Gradio app with recommendation and visualization logic.
- `models/`: Machine learning scripts and pipelines.
- `data/`: Processed book metadata, embeddings, and model outputs.
- `scripts/`: Data preprocessing and automation scripts.

---

## Testing and Validation

- Unit and integration tests covering backend APIs, ML models, and dashboard functionality.
- Performance validated to ensure under 3-second response time under typical user loads.
- Recommendations empirically validated through human evaluators with precision scores of ~80%.
- Visualization interpretability validated via PCA clusters and silhouette scores.

---

## Deployment

- Containerized using Docker for ease of local development and cloud deployment.
- Ready for deployment on AWS with monitoring of system health and API usage.
- Continuous integration pipelines configured to automate tests and deployments.

---

## Contribution & Support

Please open issues or pull requests for bugs, features, or documentation requests.

Contact Ara Sukiyan via email at asukiasy@gmail.com for project-related questions or support.

---

