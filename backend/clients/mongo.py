from typing import List, Dict, Union
from pymongo import MongoClient as PyMongoClient

from config import MONGO_DB_CONN_STR


class MongoClient:
    def __init__(self):
        self.client = PyMongoClient(MONGO_DB_CONN_STR, maxPoolSize=50)
        self.db = self.client["instagram"]
        self.collection = self.db["instagram"]

    def insert_query(self, query_text: str, results: list):
        self.collection.insert_one({"query": query_text, "results": results})

    def insert_queries(self, queries: List[Dict[str, Union[str, List]]]):
        self.collection.insert_many(queries)

    def get_all(self):
        from utils.file_utils import FileUtils

        fu = FileUtils()
        cursor = self.collection.find({})

        all_documents = []
        try:
            for document in cursor:
                all_documents.append(document)
        except Exception as e:
            print(f"An error occurred: {e}")

        return all_documents

    def get_query_results(self, query_text: str):
        result = self.collection.find_one(
            {"query": query_text}, {"_id": 0, "results": 1}
        )
        return result
