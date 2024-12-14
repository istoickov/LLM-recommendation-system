import redis

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from typing import List

from models import QueryInput, SearchResponse
from process import Process

from config import FAISS_INDEX_PATH

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; adjust to specific domains as needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

process_obj = Process()
redis_client = redis.StrictRedis(
    host="localhost", port=6379, db=0, decode_responses=True
)


@app.get("/last_queries")
def get_last_queries():
    try:
        last_queries = redis_client.lrange("queries", 0, 9)  # Get the last 10 queries
        queries = []
        for query_data in last_queries:
            query, model, option = query_data.split("|")
            queries.append({"query": query, "model": model, "option": option})
        return queries
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch queries: {str(e)}"
        )


@app.post("/search", response_model=SearchResponse)
def search_faiss_index(input_data: QueryInput):
    try:
        # Extract and validate input
        model_name = input_data.model_name
        query_text = input_data.query
        option = input_data.option

        # Preprocess the query
        preprocessed_query = process_obj.preprocess_query(query_text, option)
        if not preprocessed_query:
            raise HTTPException(
                status_code=400, detail="Invalid query after preprocessing."
            )

        # Get query embeddings
        query_embeddings = process_obj.get_query_embeddings(
            preprocessed_query, model_name
        )
        if query_embeddings is None:
            raise HTTPException(
                status_code=500, detail="Failed to generate query embeddings."
            )

        # Load the FAISS index
        index_path = FAISS_INDEX_PATH.format(option, model_name)
        index = process_obj.faiss_client.retrive_index(index_path)
        if index is None:
            raise HTTPException(
                status_code=404,
                detail=f"Index not found for model: {model_name} and option: {option}.",
            )

        # Perform the search
        results = process_obj.query_faiss_index(index, query_embeddings)
        if not results:
            raise HTTPException(status_code=404, detail="No matching results found.")

        redis_key = "queries"

        # Store the query and results in Redis (with a key)
        query_cache = f"{input_data.query}|{input_data.model_name}|{input_data.option}"
        redis_client.lpush(redis_key, query_cache)  # Push to the front of the list

        # Keep only the last 10 queries
        redis_client.ltrim(redis_key, 0, 9)

        return SearchResponse(results=results)

    except HTTPException as http_ex:
        raise http_ex

    except Exception as ex:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(ex)}"
        )
