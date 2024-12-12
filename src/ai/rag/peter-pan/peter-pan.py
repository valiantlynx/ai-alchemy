import ollama
import os
import json
from tqdm import tqdm


def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in tqdm(f.readlines()):
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((" ").join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append((" ").join(buffer))
        return paragraphs


def get_embeddings(modelname, chunks):
    return [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in tqdm(chunks)
    ]


def save_embeddings(filename, embeddings):
    # create dir if it does not exist
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    # dump embeddings to json
    with open(f"embeddings/{filename}.json") as f:
        json.dump(embeddings, f)


def main():
    # open files
    filename = "peter-pan.txt"
    paragraphs = parse_file(filename)
    embeddings = get_embeddings("llava-phi3", paragraphs[3:23])
    print(paragraphs[:2], embeddings[:2], len(embeddings))


if __name__ == "__main__":
    main()
