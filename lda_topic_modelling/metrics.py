from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim import corpora
import numpy as np
import pickle
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# --- 1. Coherence Score ---
def compute_coherence(lda_model, texts, dictionary, top_n=10):
    coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()


# --- 2. Topic Diversity ---
def compute_topic_diversity(lda_model, top_n=10):
    topic_words = []
    for topic_id in range(lda_model.num_topics):
        topic_terms = lda_model.show_topic(topic_id, topn=top_n)
        topic_words.extend([word for word, _ in topic_terms])
    unique_words = set(topic_words)
    return len(unique_words) / len(topic_words)


# --- 3. Topic Coverage ---
def compute_topic_coverage(lda_model, corpus, threshold=0.50):
    covered_docs = 0
    for doc in corpus:
        topic_distribution = lda_model.get_document_topics(doc, minimum_probability=0.0)
        max_topic_prob = max([prob for _, prob in topic_distribution])
        if max_topic_prob > threshold:
            covered_docs += 1
    return covered_docs / len(corpus)

# --- 4. Intertopic Distance Map ---
def generate_intertopic_distance_map(lda_model, corpus, dictionary, filename):
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, filename)
    print(f"Intertopic Distance Map saved as {filename}")


if __name__ == "__main__":
    # Load LDA model
    train_size = 70
    print(f"Evaluation metrics for train size : {train_size}")
    print("Loading LDA model...")
    lda_model = LdaModel.load(f"{train_size}_lda_model")

    # Load dictionary
    print("Loading dictionary...")
    dictionary = corpora.Dictionary.load(f"{train_size}_lda_dictionary.dict")

    # Load corpora
    print("Loading lda_corpus_train...")
    with open(f"{train_size}_lda_corpus_train.pkl", "rb") as f:
        corpus_train = pickle.load(f)
    print("Lodaing lda_corpus_test")
    with open(f"{train_size}_lda_corpus_test.pkl", "rb") as f:
        corpus_test = pickle.load(f)

    # Load tokenized docs
    print("Loading lda_docs_tokenized_train")
    with open(f"{train_size}_lda_docs_tokenized_train.pkl", "rb") as f:
        tokenized_train_docs = pickle.load(f)
    print("Loading lda_docs_tokenized_test")
    with open(f"{train_size}_lda_docs_tokenized_test.pkl", "rb") as f:
        tokenized_test_docs = pickle.load(f)

    # # --- Calculate Metrics ---
    # print("Calculating train coherence score")
    # train_coherence = compute_coherence(lda_model, tokenized_train_docs, dictionary)
    
    # print("Calculating test coherence score")
    # test_coherence = compute_coherence(lda_model, tokenized_test_docs, dictionary)

    # print("Calculating topic diversity score")
    # topic_diversity = compute_topic_diversity(lda_model)

    # print("Calculating train coverage score")
    # topic_coverage = compute_topic_coverage(lda_model, corpus_test)  # coverage on test data

    print("Generating the LDA Inter Topic Distance Map...")
    generate_intertopic_distance_map(lda_model, corpus_train, dictionary, f"{train_size}_lda_intertopic_distance_map.html")

    # print(f"Coherence Score on Train Data : {train_coherence:.4f}")
    # print(f"Coherence Score on Test Data  : {test_coherence:.4f}")
    # print(f"Topic Diversity Score         : {topic_diversity:.4f}")
    # print(f"Topic Coverage Score          : {topic_coverage:.4f}")
