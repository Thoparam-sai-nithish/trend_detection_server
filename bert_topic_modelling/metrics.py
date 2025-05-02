import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

def load_model_and_data(train_size):
    print("Loading BERTopic model...")
    topic_model = BERTopic.load(f"./{train_size}_bertopic_model")

    print("Loading cleaned training documents...")
    with open(f"../data/docs_train_{train_size}.pkl", "rb") as f:
        cleaned_train_docs = pickle.load(f)

    print("Loading cleaned test documents...")
    with open(f"../data/docs_test_{train_size}.pkl", "rb") as f:
        cleaned_test_docs = pickle.load(f)

    return topic_model, cleaned_train_docs, cleaned_test_docs

# Training Evaluation
def calculate_train_topic_coherence(topic_model, cleaned_docs, top_n_words=10):
    print("Calculating topic coherence on training data...")

    topics = topic_model.get_topics()
    topic_words = [
        [word for word, _ in topic_model.get_topic(t_id)[:top_n_words]]
        for t_id in topics if t_id != -1
    ]

    dictionary = Dictionary([doc.split() for doc in cleaned_docs])  # Tokenize
    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=[doc.split() for doc in cleaned_docs],
        dictionary=dictionary,
        coherence='c_v'
    )

    coherence_score = coherence_model.get_coherence()
    print(f"Topic Coherence Score on Training Data (c_v): {coherence_score:.4f}")
    return coherence_score

# Topic Diversity
def calculate_model_topic_diversity(topic_model, top_n_words=10):
    print("Calculating topic diversity on training data...")
    all_words = []
    for topic_id, topic in topic_model.get_topics().items():
        if topic_id != -1:
            all_words.extend([word for word, _ in topic[:top_n_words]])

    unique_words = set(all_words)
    diversity_score = len(unique_words) / len(all_words) if all_words else 0
    print(f"Topic Diversity Score on Training Data : {diversity_score:.4f}")
    return diversity_score

# Inter Topic Distance map
def plot_intertopic_distance(topic_model, train_size):
    print("Generating the BERTopic Inter Topic Distance Map...")
    fig = topic_model.visualize_topics()
    fig.show()
    # Save the visualization as an HTML file
    fig.write_html(f"{train_size}_bertopic_intertopic_distance_map.html")

# Test Evaluation
def calculate_test_topic_coherence(topic_model, cleaned_docs, top_n_words=10):
    print("Calculating topic coherence on test data...")

    # Tokenize the cleaned test documents
    tokenized_docs = [doc.split() for doc in cleaned_docs]

    # Create a dictionary based on the test data
    dictionary = Dictionary(tokenized_docs)

    # Extract top-n topic words
    topic_words = []
    for topic_id in topic_model.get_topics():
        if topic_id != -1:
            words = [word for word, _ in topic_model.get_topic(topic_id)[:top_n_words]]
            # Filter topic words to keep only those in the dictionary
            filtered_words = [word for word in words if word in dictionary.token2id]
            if filtered_words:  # Avoid empty lists
                topic_words.append(filtered_words)

    # Compute coherence using gensim
    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence='c_v'
    )

    coherence_score = coherence_model.get_coherence()
    print(f"Topic Coherence Score on Test Data (c_v): {coherence_score:.4f}")
    return coherence_score

def calculate_bertopic_coverage(topic_model, test_docs):
    print("Calculating Topic Coverage for BERTopic on Test Data...")
    predicted_topics, _ = topic_model.transform(test_docs)
    covered_docs = sum(1 for topic in predicted_topics if topic != -1)
    coverage_ratio = covered_docs / len(predicted_topics)
    print(f"BERTopic Coverage on Test Data: {coverage_ratio:.2%}")
    return coverage_ratio

if __name__ == "__main__":
    train_size = 90
    topic_model, cleaned_train_docs, cleaned_test_docs = load_model_and_data(train_size)
    # train_coherence = calculate_train_topic_coherence(topic_model, cleaned_train_docs)
    # test_coherence = calculate_test_topic_coherence(topic_model, cleaned_test_docs)
    # topic_diversity = calculate_model_topic_diversity(topic_model)
    # bertopic_coverage = calculate_bertopic_coverage(topic_model, cleaned_test_docs)
    plot_intertopic_distance(topic_model, train_size)

    # print(f"Coherence Score on Train Data : {train_coherence:.4f}")
    # print(f"Coherence Score on Test Data : {test_coherence:.4f}")
    # print(f"Topic Diversity Score : {topic_diversity:.4f}")
    # print(f"Topic Coverage Score : {bertopic_coverage:.4f}")
