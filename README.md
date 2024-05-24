# **Movie and TV Show Semantic Search Application**
This application enables users to search for movies and TV shows based on their titles, overviews, genres, and directors using Elasticsearch and a pre-trained Sentence Transformer model. The application provides a simple user interface powered by Streamlit.

## Features
- **Semantic Search:** Utilizes Elasticsearch and a Sentence Transformer model to perform semantic search on a dataset of movies and TV shows.
- **Cosine Similarity:** Calculates cosine similarity between the user's query and the overview of each document to rank search results.
- **Streamlit UI:** Offers a user-friendly interface with a text input field for entering search queries and a search button for initiating searches.

## Prerequisites
Before running the application, make sure you have the following dependencies installed:

- Python 3.x
- Elasticsearch
- Streamlit
- Sentence Transformer
- Pandas
- NumPy
- Kaggle API (for dataset download)

You can install Python dependencies using pip:

```bash
pip install streamlit sentence-transformers elasticsearch pandas numpy kaggle
```

## Usage
1. Navigate to Project Directory:

```bash
cd Assignment 2
```

2. Run the Streamlit app:

```bash
streamlit run search_app.py
```
3. Access the application in your web browser by opening the provided URL (usually `http://localhost:8501`)

4. Enter your search query in the text input field and click the "Search" button to retrieve relevant Movies and TV Shows.

## Data Source
The dataset used in this application is obtained from Kaggle: [IMDb Dataset of Top 1000 Movies and TV Shows](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows).

## References
1. https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic-search/semantic_search_quora_elasticsearch.py

2. https://github.com/ev2900/Cosine_Similarity_Search_Example

3. https://sbert.net/docs/pretrained_models.html

## Credits
- Author: Shloka Bhatt
- Student ID: 1002150636
