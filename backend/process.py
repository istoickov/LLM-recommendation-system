import json

import numpy as np

from backend.consts.models import BERT, MINILM, ROBERTA, SBERT, SBERT_SHORT
from clients.mongo import MongoClient
from clients.redis import RedisClient
from clients.faiss import FaissClient

from utils.data_utils import DataUtils
from utils.file_utils import FileUtils

from utils.ml_utils import MachineLearningUtils

from models import SearchResult, InstagramData

from config import (
    PREPROCESSED_DATA,
    # EMBEDDINGS_PATH,
    FAISS_INDEX_PATH,
)


class Process:
    def __init__(self) -> None:
        self.redis_client = RedisClient()
        self.mongo_client = MongoClient()
        self.faiss_client = FaissClient()

        self.data_utils = DataUtils()
        self.file_utils = FileUtils()
        self.ml_utils = {
            "bert": MachineLearningUtils(BERT),
            "sbert": MachineLearningUtils(SBERT),
            "roberta": MachineLearningUtils(ROBERTA),
            "minilm": MachineLearningUtils(MINILM),
            "sbert_short": MachineLearningUtils(SBERT_SHORT),
        }

    def save_data(self, data, file_name=PREPROCESSED_DATA):
        self.file_utils.write_file(file_name, data)

    def load_data(self, file_name=PREPROCESSED_DATA):
        return self.file_utils.read_file(file_name)

    def make_embeddings(self, data, model_name, local=False):
        model = self.ml_utils[model_name]

        embeddings = []
        for item in data:
            item_embeddings = model.get_embeddings(item)
            embeddings.append(item_embeddings)
        return np.array(embeddings)

    def get_query_embeddings(self, query, model_name):
        query_embeddings = self.ml_utils[model_name].get_query_embeddings(query)
        if len(query_embeddings.shape) == 3:
            query_embeddings = query_embeddings[:, 0, :]
        return query_embeddings

    def preprocess_query(self, query, option):
        if option == 1:
            query = self.data_utils.lemmatization_senetence(query)
        elif option == 2:
            query = self.data_utils.stemm_sentence(query)
        elif option == 3:
            query = self.data_utils.remove_stopwords(query)
        elif option == 4:
            query = self.data_utils.remove_stopwords([query])
            query = self.data_utils.stemm_sentence(query[0])
        elif option == 5:
            query = self.data_utils.remove_stopwords([query])
            query = self.data_utils.lemmatization_senetence(query[0])

        return query

    def make_faiss_index(self, embedings, name):
        index = self.faiss_client.create_index(embedings)
        self.faiss_client.save_index(index, FAISS_INDEX_PATH.format(name))
        return index

    def query_faiss_index(self, index, query_embeddings):
        original_data = json.load(open("./data/_model_data.json", "r"))
        distances, indices = index.search(query_embeddings, 10)

        results = []
        for index, distance in zip(indices[0], distances[0]):
            item = original_data[index]
            tags = [tag["description"] for tag in item.get("tags", [])]
            results.append(
                SearchResult(
                    index=index,
                    distance=distance,
                    instagram_data=InstagramData(
                        name=item["name"],
                        country=item["state"],
                        full_name=item.get("instagram", {}).get("full_name"),
                        bio=item.get("instagram", {}).get("bio"),
                        follows=item.get("instagram", {}).get("follows"),
                        following=item.get("instagram", {}).get("following"),
                        tags=tags,
                    ),
                )
            )
        return results
