import os
import json
import torch
import shutil
import gc
from transformers import BertTokenizerFast, BertForMaskedLM, AutoConfig
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
from concurrent.futures import ThreadPoolExecutor

from consts.models import SBERT, ROBERTA, MSMARCO, MINILM
from config import (
    GENERATED_QUERY_FILE,
    GENERATED_QUERY_FOLDER,
    DATA_FINE_TUNE_FOLDER,
    PREPROCESSED_DATA_OPTION,
)

DEBUG = False

MODELS_TO_PROCESS = [
    MSMARCO,
    SBERT,
    ROBERTA,
    MINILM,
]


def is_mlm_compatible(model_name):
    """
    Check if the model supports Masked Language Modeling (MLM).
    """
    try:
        config = AutoConfig.from_pretrained(model_name)
        return "MaskedLM" in config.architectures
    except Exception as e:
        print(f"Could not determine MLM compatibility for {model_name}: {e}")
        return False


def synthetic_query_generation(summaries, model_name, option):
    """
    Generate synthetic queries using a model if it supports MLM.
    """
    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        mask_token = tokenizer.mask_token
        if not mask_token:
            print(f"No [MASK] token available for {model_name}. Skipping.")
            return

        model = BertForMaskedLM.from_pretrained(model_name).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        scaler = GradScaler()  # Enable mixed precision
    except Exception as e:
        print(f"Error initializing tokenizer or model for {model_name}: {e}")
        return

    # Parameters
    batch_size = 8
    max_length_paragraph = min(512, tokenizer.model_max_length)

    def _remove_non_ascii(s):
        return "".join(i for i in s if ord(i) < 128)

    folder = GENERATED_QUERY_FOLDER.format(option, model_name.replace("/", "-"))
    os.makedirs(folder, exist_ok=True)
    generated_queries_file_path = os.path.join(folder, GENERATED_QUERY_FILE)
    with open(generated_queries_file_path, "a") as generated_queries_file:
        for start_idx in tqdm(range(0, len(summaries), batch_size)):
            sub_paragraphs = summaries[start_idx : start_idx + batch_size]
            for para in sub_paragraphs:
                para_tokens = tokenizer.tokenize(para)
                if len(para_tokens) > max_length_paragraph:
                    para_chunks = [
                        para_tokens[i : i + max_length_paragraph]
                        for i in range(0, len(para_tokens), max_length_paragraph)
                    ]
                else:
                    para_chunks = [para_tokens]

                for chunk in para_chunks:
                    chunk.insert(len(chunk) // 2, mask_token)
                    masked_text = tokenizer.convert_tokens_to_string(chunk)

                    inputs = tokenizer(
                        masked_text,
                        max_length=max_length_paragraph,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    ).to(model.device)

                    with autocast():
                        outputs = model(**inputs)

                    predictions = outputs.logits.argmax(dim=-1)
                    predicted_tokens = tokenizer.convert_ids_to_tokens(predictions[0])
                    predicted_query = tokenizer.convert_tokens_to_string(
                        predicted_tokens
                    )

                    para_clean = _remove_non_ascii(para).replace("\t", " ").strip()
                    predicted_query = (
                        _remove_non_ascii(predicted_query).replace("\t", " ").strip()
                    )
                    generated_queries_file.write(f"{predicted_query}\t{para_clean}\n")

    torch.cuda.empty_cache()
    gc.collect()
    print(f"GPU memory cleared after synthetic query generation for {model_name}.")


def make_model_with_mean_pooling(
    model_name, option, skip_generation=False, summaries=None
):
    """
    Fine-tune a SentenceTransformer with mean pooling.
    """
    try:
        word_emb = models.Transformer(model_name)
        pooling = models.Pooling(word_emb.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_emb, pooling]).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    except Exception as e:
        print(f"Skipping fine-tuning for {model_name}: {e}")
        return

    train_examples = []
    if not skip_generation:
        try:
            folder = GENERATED_QUERY_FOLDER.format(option, model_name.replace("/", "-"))
            generated_queries_file_path = os.path.join(folder, GENERATED_QUERY_FILE)
            with open(generated_queries_file_path, "r") as fIn:
                for line in fIn:
                    try:
                        query, paragraph = line.strip().split("\t", maxsplit=1)
                        train_examples.append(InputExample(texts=[query, paragraph]))
                    except Exception as e:
                        print(f"Skipping invalid line: {e}")
        except Exception as e:
            print(
                f"Error loading synthetic data for {model_name}, option {option}: {e}"
            )
    else:
        if summaries is None:
            print(
                f"Summaries are required for direct training when skipping query generation."
            )
            return
        for summary in summaries:
            train_examples.append(InputExample(texts=[summary, summary]))

    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=4, num_workers=2
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)
    num_epochs = 1
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        use_amp=True,  # Mixed Precision for SentenceTransformers
    )

    model_save_path = f"{DATA_FINE_TUNE_FOLDER}/{model_name.replace('/', '-')}-{option}"
    os.makedirs(DATA_FINE_TUNE_FOLDER, exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    torch.cuda.empty_cache()
    gc.collect()
    print(f"GPU memory cleared after fine-tuning {model_name}.")


def make_my_models():
    models_to_process = MODELS_TO_PROCESS

    with ThreadPoolExecutor(max_workers=2) as executor:
        for model_name in models_to_process:
            is_mlm = is_mlm_compatible(model_name)
            print(f"Processing model: {model_name} (MLM Compatible: {is_mlm})")

            option = 0  # Only process option 0
            try:
                with open(PREPROCESSED_DATA_OPTION.format(option), "r") as f:
                    cleaned_text_summaries = json.load(f)
            except Exception as e:
                print(f"Error loading data for option {option}: {e}")
                continue

            if DEBUG:
                cleaned_text_summaries = cleaned_text_summaries[:10]
            else:
                print(
                    f"Total summaries for option {option}: {len(cleaned_text_summaries)}"
                )

            if is_mlm:
                executor.submit(
                    synthetic_query_generation,
                    cleaned_text_summaries,
                    model_name,
                    option,
                )
            make_model_with_mean_pooling(
                model_name,
                option,
                skip_generation=not is_mlm,
                summaries=cleaned_text_summaries,
            )

    print("Compressing models...")
    shutil.make_archive("fine_tuned_models", "zip", "./data_fine_tune")
    print("Models compressed into fine_tuned_models.zip")


if __name__ == "__main__":
    make_my_models()
    print("Download the models locally: fine_tuned_models.zip")
