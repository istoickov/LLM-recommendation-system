# Use Docker container name for MongoDB
# TODO put env variables in .envrc file
MONGO_DB_CONN_STR = "mongodb://localhost:27017/"

RAW_DATA = "./data/_model_data.json"

PREPROCESSED_DATA = "./data/preprocessed_text.json"
PREPROCESSED_DATA_MULTIPLE_STRINGS = "./data/preprocessed_text_multiple_strings.json"
PREPROCESSED_DATA_OPTION = "./data/data_{}.json"

EMBEDDINGS_PATH = "./data/{}_{}_{}.npy"

FAISS_INDEX_PATH = "./data/faiss_index_{}_{}.faiss"

TOP_RESULTS_FILE_PATH = "./data/top_results-{}_average_scores.json"

EVALUATION_RESULTS = "./evaluation_results.json"

GENERATED_QUERY_FILE = "generated_queries_all.tsv"
GENERATED_QUERY_FOLDER = "./data_fine_tune/option_{}/{}/"

DATA_FINE_TUNE_FOLDER = "./data_fine_tune/"
