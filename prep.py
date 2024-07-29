import os
import requests
import struct
from tqdm import tqdm
import tiktoken
import numpy as np
import torch

DATA_CACHE_DIR = "data"
enc = tiktoken.get_encoding("gpt2")


def encode(s):
    return enc.encode(s, allowed_special={'<|endoftext|>'})


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download():
    """Downloads the TinyShakespeare dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare.txt")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")


def tokenize():
    data_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare.txt")
    text = open(data_filename, 'r').read()
    text = "<|endoftext|>" + text
    text = text.replace('\n\n', '\n\n<|endoftext|>')
    tokens = encode(text)
    tokens_np = np.array(tokens, dtype=np.int32)
    val_tokens_np = tokens_np[:32768]
    train_tokens_np = tokens_np[32768:]
    val_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_val.bin")
    train_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_train.bin")
    with open(val_filename, "wb") as f:
        f.write(val_tokens_np.tobytes())
    with open(train_filename, "wb") as f:
        f.write(train_tokens_np.tobytes())
    print(f"Saved {len(val_tokens_np)} tokens to {val_filename}")
    print(f"Saved {len(train_tokens_np)} tokens to {train_filename}")


def write_tokenizer(enc, filename):
    n = enc.max_token_value + 1
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240328  # magic
    header[1] = 2  # tokenizer version = 2 (1 -> 2: includes EOT token)
    header[2] = n  # number of tokens
    header[3] = enc.eot_token  # EOT token
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes())
        for i in range(n):
            b = enc.decode_bytes([i])
            length = len(b)
            assert length < 256, f"Token length exceeds 255: {length}"
            file.write(struct.pack("<B", length))
            file.write(b)
    print(f"Saved tokenizer information to {filename}")


if __name__ == "__main__":
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    download()
    tokenize()

    tokenizer_filename = os.path.join(DATA_CACHE_DIR, "gpt2_tokenizer.bin")
    write_tokenizer(enc, tokenizer_filename)
