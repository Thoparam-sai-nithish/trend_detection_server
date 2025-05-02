import os, sys
import random
import codecs
import pickle
from gensim.models.ldamodel import LdaModel
from gensim import corpora
from nltk.corpus import stopwords 
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)
from utils import clean_doc

print("Start")

def clean(doc):
    return list(clean_doc(doc).split()) 

def prepare_data(cleaned_docs, dictionary=None):
    tokenized_docs = [doc.split() for doc in cleaned_docs]
    
    if dictionary is None:
        dictionary = corpora.Dictionary(tokenized_docs)
        dictionary.filter_extremes(no_below=4, no_above=0.4)

    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    return dictionary, corpus, tokenized_docs

def train_lda_model(dictionary, corpus, num_topics=50, passes=50, iterations=500):
    lda = LdaModel(corpus=corpus,
                   id2word=dictionary,
                   num_topics=num_topics,
                   random_state=42,
                   passes=passes,
                   iterations=iterations,
                   alpha='auto')
    return lda

def save_artifacts(lda_model, dictionary, train_corpus, test_corpus,
                   train_tokenized_docs, test_tokenized_docs, train_size):
    lda_model.save(f"./{train_size}_lda_model")
    dictionary.save(f"./{train_size}_lda_dictionary.dict")

    # Save train corpus & tokens
    with open(f"./{train_size}_lda_corpus_train.pkl", "wb") as f:
        pickle.dump(train_corpus, f)
    with open(f"./{train_size}_lda_docs_tokenized_train.pkl", "wb") as f:
        pickle.dump(train_tokenized_docs, f)

    # Save test corpus & tokens
    with open(f"./{train_size}_lda_corpus_test.pkl", "wb") as f:
        pickle.dump(test_corpus, f)
    with open(f"./{train_size}_lda_docs_tokenized_test.pkl", "wb") as f:
        pickle.dump(test_tokenized_docs, f)

    # Save topic summary
    with open(f"./{train_size}_lda_topic_summary.txt", "w", encoding="utf-8") as f:
        for i in range(lda_model.num_topics):
            words = lda_model.show_topic(i, topn=10)
            f.write(f"Topic {i}: {', '.join([w for w, _ in words])}\n")

    print("✅ LDA model and all artifacts saved!")

if __name__ == "__main__":
    # Load cleaned train & test docs
    train_size = 70
    train_data_path = f"../data/docs_train_{train_size}.pkl"
    test_data_path = f"../data/docs_test_{train_size}.pkl"

    print("Loading the training data...")
    with open(train_data_path, "rb") as f:
        cleaned_train_docs = pickle.load(f)

    print("Loading the testing data..")
    with open(test_data_path, "rb") as f:
        cleaned_test_docs = pickle.load(f)

    print("Preparing training data...")
    dictionary, train_corpus, train_tokenized_docs = prepare_data(cleaned_train_docs)

    print("Preparing test data using training dictionary...")
    _, test_corpus, test_tokenized_docs = prepare_data(cleaned_test_docs, dictionary)

    print("Training LDA model...")
    lda_model = train_lda_model(dictionary, train_corpus, num_topics=50)

    print("Saving artifacts...")
    save_artifacts(lda_model, dictionary, train_corpus, test_corpus,
                   train_tokenized_docs, test_tokenized_docs, train_size)
 