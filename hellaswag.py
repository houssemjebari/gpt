import requests
import tiktoken 
import tqdm
import time
import os 

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


if __name__ == "__main__": 
    split='val'
    download(split)