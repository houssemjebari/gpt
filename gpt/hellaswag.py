import requests
import tiktoken 
import torch
import tqdm
import time
import os 
import json

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__),"hellaswag")

hellaswags = {
    'train': "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    'val': "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    'test': 'https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl'
}

enc = tiktoken.get_encoding('gpt2')


def download(split):
    ''' Download HellaSwag Data'''
    assert split in ["train", "val", "test"], ValueError("make sure the split provided is either: train, val or test")
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)


def download_file(url, filename, chunk_size=1024, retries=3, delay=5):
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt + 1} to download {url}")
            resp = requests.get(url, stream=True, timeout=10)
            resp.raise_for_status()  # Raise an HTTPError for bad responses
            total = int(resp.headers.get("Content-Length", 0))
            with open(filename, "wb") as file:
                for data in resp.iter_content(chunk_size=chunk_size):
                    file.write(data)
            print(f"Downloaded {filename} successfully!")
            return  # Exit function if successful
        except requests.exceptions.RequestException as e:
            print(f"Download error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)  # Wait before retrying
        except OSError as e:
            print(f"File error: {e}. Retrying might not help.")
            break  # Exit loop for non-recoverable errors
    print(f"Failed to download {url} after {retries} attempts.")

def render_example(example):
    # Extract the data
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    token_rows = []
    mask_rows = []
    # Tokenize the ctx
    ctx = enc.encode(ctx)
    for ending in endings:
        ending = enc.encode(ending)
        token_rows.append(ctx + ending)
        mask_rows.append([0] * len(ctx) + [1] * len(ending))

    max_len = max([len(row) for row in token_rows])
    tokens = torch.zeros((4,max_len), dtype=torch.long)
    masks = torch.zeros((4,max_len), dtype=torch.long)
    for i, (token_row, mask_row) in enumerate(zip(token_rows, mask_rows)):
        tokens[i,:len(token_row)] = torch.tensor(token_row,dtype=torch.long)
        masks[i,:len(mask_row)] = torch.tensor(mask_row,dtype=torch.long)

    return tokens, masks, label

def iterate_examples(split):
    download(split)
    with open(os.path.join(DATA_CACHE_DIR,f"hellaswag_{split}.jsonl")) as f:
        for line in f:
            example = json.loads(line)
            yield example

if __name__ == "__main__": 
    split='val'
    download(split)