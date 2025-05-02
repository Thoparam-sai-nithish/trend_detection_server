import os, sys, random, codecs, pickle
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)
from utils import clean_doc

def load_docs(corpus_path, train_size, seed=42):
    article_paths = [os.path.join(corpus_path, p) for p in os.listdir(corpus_path)]
    random.seed(seed)
    random.shuffle(article_paths)
    print(f"Loading documents from {corpus_path}")

    doc_complete = []
    for path in article_paths:
        with codecs.open(path, 'r', 'utf-8') as fp:
            doc_content = fp.read()
            doc_complete.append(doc_content)

    split_index = int(len(doc_complete) * train_size)
    docs_train = doc_complete[:split_index]
    docs_test = doc_complete[split_index:]
    return docs_train, docs_test


def clean_all_docs(docs_all):
    return [clean_doc(doc) for doc in docs_all]



if __name__ == "__main__":
    corpus_path = "./articles-corpus"
    train_size = 0.70

    docs_train, docs_test = load_docs(corpus_path, train_size)
    print("Cleaning datasets...")
    clean_train = clean_all_docs(docs_train)
    clean_test = clean_all_docs(docs_test)

    print(f"Train Files: {len(clean_train)}")
    print(f"Test  Files: {len(clean_test)}")



    with open(f"docs_train_{int(train_size*100)}.pkl", 'wb') as f:
        pickle.dump(clean_train, f)

    with open(f"docs_test_{int(train_size*100)}.pkl", 'wb') as f:
        pickle.dump(clean_test, f)

    print(f"Documents saved to docs_train_{int(train_size*100)}.pkl and docs_test_{int(train_size*100)}.pkl ✅")
