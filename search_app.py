import streamlit as st # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
from elasticsearch import Elasticsearch # type: ignore
import numpy as np # type: ignore

indexName = "my_movies"

# Initialize Elasticsearch client
try:
    es = Elasticsearch(
        "http://localhost:9200",
        http_auth=("shloka","shloka"),
        ca_certs="/Users/shlok/Downloads/elasticsearch-8.13.0-windows-x86_64/elasticsearch-8.13.0/config/certs/http_ca.crt"
    )
except ConnectionError as e:
    print("Connection Error: ", e)

if es.ping():
    print("Successfully connected to Elasticsearch!")
else:
    print("Failed to connect to Elasticsearch!")

def calculate_cosine_similarity(input_vector, target_vector):
        return cosine_similarity([input_vector], [target_vector])[0][0]

def search(input_keyword):
    model = SentenceTransformer("all-mpnet-base-v2")
    vectorOfInputKeyword = model.encode(input_keyword)

    res = es.search(
        index="my_movies",
        body={
            "knn": {
                "field": "Overview_Vector",
                "query_vector": vectorOfInputKeyword,
                "k": 10,
            }
        }
    )
    hits = res["hits"]["hits"]

    hits_sorted = sorted(hits, key=lambda x: calculate_cosine_similarity(vectorOfInputKeyword, x["_source"]["Overview_Vector"]), reverse=True)

    # Calculate cosine similarity and add it to the results
    for hit in hits_sorted:
        if '_source' in hit:
            vectorOfDocument = model.encode(hit['_source']['Overview'])
            cosine_similarity = np.dot(vectorOfInputKeyword, vectorOfDocument) / (np.linalg.norm(vectorOfInputKeyword) * np.linalg.norm(vectorOfDocument))
            hit['_score'] = cosine_similarity

    # Sort results based on cosine similarity in descending order
    hits.sort(key=lambda x: x['_score'], reverse=True)

    return hits

# Streamlit app
def main():
    st.markdown(
        """
        <style>
        body {
            background-image: url("https://www.vecteezy.com/vector-art/6915132-minimalist-style-hand-painted-liquid-background");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            font-family: Arial, sans-serif;
            color: white;  /* Change font color to white */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Vector Search App")
    # Get user input
    query = st.text_input("Enter your query:")

    if st.button("Search"):
        if query:
            results = search(query)
            st.subheader("Search Results")
            for result in results:
                with st.container():
                    if '_source' in result:
                        try:
                            st.header(f"{result['_source']['Series_Title']}")
                        except Exception as e:
                            print(e)
                        
                        try:
                            st.write(f"Overview: {result['_source']['Overview']}")
                        except Exception as e:
                            print(e)
                        
                        try:
                            st.write(f"Director: {result['_source']['Director']}")
                        except Exception as e:
                            print(e)
                        st.divider()

if __name__ == "__main__":
    main()
