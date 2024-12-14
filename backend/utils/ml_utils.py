import os
import numpy as np
import torch

from tqdm import tqdm

from consts.models import BERT, ROBERTA, DISTILBERT, MSMARCO
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    models,
    datasets,
)
from transformers import AutoTokenizer, AutoModelForMaskedLM


from sentence_transformers import (
    SentenceTransformer,
)


HUGGING_FACE_TOKEN = "<HUGGING_FACE_TOKEN>"


def syntethic_query_generation(summaries, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(
        "mps" if torch.cuda.is_available() else "cpu"
    )

    # Parameters for generation
    batch_size = 4  # Reduced batch size for better memory management
    num_queries = 5  # Number of queries to generate for every paragraph
    max_length_paragraph = 512  # Max length for paragraph
    max_length_query = 64  # Max length for output query

    def _removeNonAscii(s):
        return "".join(i for i in s if ord(i) < 128)

    # Open the output file once for all iterations (to avoid IO overhead)
    with open(
        "../data_fine_tune/generated_queries_all.tsv", "a"
    ) as generated_queries_file:
        for start_idx in tqdm(range(0, len(summaries), batch_size)):
            sub_paragraphs = summaries[start_idx : start_idx + batch_size]
            inputs = tokenizer.prepare_seq2seq_batch(
                sub_paragraphs,
                max_length=max_length_paragraph,
                truncation=True,
                return_tensors="pt",
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_length=max_length_query,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=num_queries,
                no_repeat_ngram_size=2,  # Avoid repeating n-grams in the output
            )

            # Process and write the results in batches to the file
            for idx, out in enumerate(outputs):
                query = tokenizer.decode(out, skip_special_tokens=True)
                query = _removeNonAscii(query)

                para = sub_paragraphs[int(idx / num_queries)]
                para = _removeNonAscii(para)

                generated_queries_file.write(
                    f"{query.replace('\t', ' ').strip()}\t{para.replace('\t', ' ').strip()}\n"
                )


def make_model_with_mean_pooling(model_name):
    # Create the model with mean pooling
    word_emb = models.Transformer(model_name)
    pooling = models.Pooling(word_emb.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_emb, pooling]).to(
        "mps" if torch.cuda.is_available() else "cpu"
    )

    # Prepare the training examples
    train_examples = []
    with open("../data_fine_tune/generated_queries_all.tsv") as fIn:
        for line in fIn:
            try:
                query, paragraph = line.strip().split("\t", maxsplit=1)
                train_examples.append(InputExample(texts=[query, paragraph]))
            except:
                pass

    # Reduce the batch size for better memory handling
    train_dataloader = datasets.NoDuplicatesDataLoader(
        train_examples, batch_size=4
    )  # Smaller batch size

    # Use MultipleNegativesRankingLoss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Set number of epochs and warmup steps
    num_epochs = 3
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

    # Fine-tune the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=False,  # Disable progress bar for faster performance
    )

    # Save the model
    os.makedirs("../data_fine_tune/", exist_ok=True)
    model.save(f"../data_fine_tune/{model_name.split('/')[-1]}")
    return model


class MachineLearningUtils:
    def __init__(self, model_name=MSMARCO):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=HUGGING_FACE_TOKEN
        )
        if model_name.startswith("models/sentence-transformers"):
            self.model = SentenceTransformer(model_name)
        elif model_name in [BERT, ROBERTA]:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        elif model_name == DISTILBERT:
            self.model = SentenceTransformer(DISTILBERT)
        else:
            self.model = SentenceTransformer(model_name, token=HUGGING_FACE_TOKEN)

    def get_embeddings(self, sentences):
        embeddings = self.model.encode(sentences)
        return np.array(embeddings, dtype=np.float32).tolist()

    def get_query_embeddings(self, search_text):
        search_vector = self.model.encode(search_text).astype("float32")
        _vector = np.array([search_vector])
        if len(_vector.shape) == 1:
            _vector = np.reshape(_vector, (1, -1))
        return _vector
