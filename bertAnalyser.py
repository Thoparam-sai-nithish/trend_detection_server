from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
import nltk
from umap import UMAP
import re

nltk.download('wordnet')

# New sample statements
statements = [
    "AI doesn't have to be able to destroy humanity. If AI has a goal and humanity just happens to be in the way, it will destroy humanity as a matter of course, without even thinking about it — no hard feelings.",
    "Believes that AI will do enormous good, but tonight he has a warning. He says that AI systems may be more intelligent than we know, and there's a chance the machines could take over, which made us ask the question.",
    "The most important thing that entrepreneurs should do is pick something they care about and work on it, but don't actually commit to turning it into a company until it's working. If you look at the data, the very best companies that have been built have done so this way, not from people who decided upfront that they wanted to start a company."
]

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Advanced Preprocessing: remove special chars, stop words, and lemmatize
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special chars
    words = text.split()
    cleaned_words = [
        lemmatizer.lemmatize(word.lower())
        for word in words if word.lower() not in ENGLISH_STOP_WORDS
    ]
    return ' '.join(cleaned_words)

cleaned_texts = [preprocess_text(statement) for statement in statements]

# Load Sentence-BERT for contextual embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Keep n_components=1 but tweak neighbors and min_dist for better separation
umap_model = UMAP(n_neighbors=10, n_components=1, min_dist=0.01, metric='cosine')

# Configure BERTopic
bertopic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    n_gram_range=(1, 2),
    min_topic_size=2,
    verbose=True,
    calculate_probabilities=True
)

# Fit BERTopic
topics, _ = bertopic_model.fit_transform(cleaned_texts)

# Configure KeyBERT
keybert_model = KeyBERT(embedding_model)

# Refined topic categories
topic_keywords = {
    "Artificial Intelligence & Machine Learning": ["ai", "machine", "learning", "intelligent"],
    "Ethics & Safety in AI": ["humanity", "warning", "safety", "goal"],
    "Entrepreneurship & Startups": ["entrepreneur", "company", "startup", "business"],
}

# Extract refined topics
unique_topics = {}

for text, statement, topic in zip(cleaned_texts, statements, topics):
    # Get BERTopic topics
    bertopic_words = bertopic_model.get_topic(topic)
    bertopic_labels = set(word for word, _ in bertopic_words if word)
    # Get KeyBERT keywords
    keybert_keywords = keybert_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=5)
    keybert_labels = set(word for word, _ in keybert_keywords)
    # Combine topics and filter only relevant ones
    combined_topics = bertopic_labels.union(keybert_labels)
    # Use strict keyword-based filtering
    refined_topics = set()
    for category, keywords in topic_keywords.items():
        if any(keyword in combined_topics for keyword in keywords):
            refined_topics.add(category)
    unique_topics[statement] = list(refined_topics)
# Output unique and contextually relevant topics for each statement
for statement, topics in unique_topics.items():
    print(f"\n📝 Statement: {statement}")
    print(f"🔍 Unique Contextual Topics: {topics}")
# Calculate and display topic frequencies
topic_frequencies = {}
for topics in unique_topics.values():
    for topic in topics:
        topic_frequencies[topic] = topic_frequencies.get(topic, 0) + 1
print("\n📊 Topic Frequencies:")
for topic, frequency in topic_frequencies.items():
    print(f"{topic}: {frequency}")