import os, time, shutil, string
from nltk.corpus import stopwords


def rmRecur(directory_path):
    if os.path.exists(directory_path):
        time.sleep(2) 
        shutil.rmtree(directory_path)
        print(f"🚮 Deleted directory: {directory_path}")

# Build Stop Words
custom_stop_words = {"a", "an", "the", "i", "me", "my", "myself", "mine", "you", "your", 
                     "yourself", "yours", "he", "him", "his", "himself", "she", "her", 
                     "hers", "herself", "it", "its", "itself", "we", "us", "our", "ours", 
                     "ourselves", "they", "them", "their", "theirs", "themselves", "this", 
                     "that", "these", "those", "who", "whom", "whose", "which", "what", 
                     "whichever", "whoever", "whomever", "whatever", "and", "or", "but", 
                     "nor", "for", "yet", "so", "about", "above", "across", "after", 
                     "against", "along", "amid", "among", "around", "as", "at", "before", 
                     "behind", "below", "beneath", "beside", "between", "beyond", "by", 
                     "despite", "down", "during", "except", "from", "in", "inside", "into", 
                     "like", "near", "of", "off", "on", "onto", "out", "outside", "over", 
                     "past", "per", "since", "through", "throughout", "to", "toward", 
                     "under", "underneath", "until", "up", "upon", "with", "within", 
                     "without", "am", "is", "are", "was", "were", "be", "being", "been", 
                     "have", "has", "had", "having", "do", "does", "did", "doing", "will", 
                     "would", "shall", "should", "can", "could", "may", "might", "must", 
                     "be able to", "be going to", "be supposed to", "be used to", "ought to", 
                     "again", "almost", "already", "also", "always", "ever", "never", 
                     "now", "often", "once", "only", "quite", "rather", "really", "seldom", 
                     "sometimes", "soon", "then", "there", "too", "very", "well", "yet", 
                     "no", "not", "neither", "none", "nor", "nothing", "nowhere", "nobody", "n't",  
                     "who", "what", "when", "where", "why", "how", "this", "that", "these", 
                     "those", "all", "any", "both", "each", "either", "enough", "every", 
                     "few", "many", "more", "most", "much", "none", "some", "several", 
                     "i'm", "you're", "he's", "she's", "it's", "we're", "they're", "i've", 
                     "you've", "we've", "they've", "isn't", "aren't", "wasn't", "weren't", 
                     "hasn't", "haven't", "hadn't", "doesn't", "don't", "didn't", "won't", 
                     "wouldn't", "can't", "couldn't", "shouldn't", "mightn't", "mustn't", 
                     "because", "if", "just", "so", "than", "though", "through", "until", 
                     "while", "with"}
custom_stop_words = {word.lower() for word in custom_stop_words}
nltk_stop_words = {word.lower() for word in stopwords.words('english')}

stop_words = custom_stop_words.union(nltk_stop_words)

punctuation_marks = set(string.punctuation)


