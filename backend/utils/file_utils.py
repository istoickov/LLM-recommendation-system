import json

import numpy as np


class FileUtils:
    @staticmethod
    def read_file(file_name):
        with open(file_name, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data

    @staticmethod
    def write_file(file_name, data):
        with open(file_name, "w", encoding="utf-8") as file:
            json.dump(data, file)

    @staticmethod
    def write_numpy_file(file_name, data):
        with open(file_name, "wb") as file:
            np.save(file, data)
    
    @staticmethod
    def write_numpy_file_generator(file_path, generator, batch_size=100):
        # Open the file for appending in binary write mode
        with open(file_path, 'ab') as f:
            batch = []
            for item in generator:
                batch.append(item)
                # When we reach batch_size, save the batch and reset it
                if len(batch) == batch_size:
                    np.save(f, np.array(batch))
                    batch = []
            # Save any remaining items in the last incomplete batch
            if batch:
                np.save(f, np.array(batch))

    @staticmethod
    def read_numpy_file(file_name):
        with open(file_name, "rb") as file:
            embeddings = np.load(file)
        return embeddings
