import pickle
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import LdaModel
from operator import itemgetter
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)
from utils import clean_doc

stop = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

current_path = os.path.dirname(__file__)
train_size = 70
lda_fp = os.path.join(current_path, f"{train_size}_lda_model")
ldamodel = LdaModel.load(os.path.join(lda_fp))

def rem_ascii(s):
    return "".join([c for c in s if ord(c) < 128 ])

def clean(doc):
    cleaned_doc = clean_doc(doc)
    x = cleaned_doc.split()
    y = [s for s in x if len(s) > 2]
    return y

def get_theme(doc):
    topics = [
        "American Cities and Sports",
        "Space and Natural Environment",
        "California and Energy/Legal Affairs",
        "Numerical Order and Ranking",
        "British Monarchy and History",
        "Biology and Animal Species",
        "Italy and Naval/Military History",
        "U.S. States and Administrative Divisions",
        "French Administrative Geography",
        "Video Games and Consoles",
        "Short Stories and News Reports",
        "Historical Timeline (Centuries)",
        "Music Albums and Bands",
        "Elements and Natural Forces/Biology",
        "Rivers and Geographical Features",
        "Historical Figures and Institutions",
        "Awards and Broadcasting",
        "Historical Dates and Figures",
        "Ancient Civilizations",
        "Global Geography and Laws",
        "American Literature and Arts",
        "Political and Religious Leadership",
        "Music Awards and Composers",
        "City and Urban Geography",
        "Australian Geography",
        "Football and Sports Teams",
        "Life Events and Mortality",
        "Wikipedia Maintenance and Biological Concepts",
        "Disney and Chemistry",
        "Common Concepts and Social Behavior",
        "Scientific Research and Studies",
        "Biology and Sports Entertainment",
        "Film and Television Industry",
        "Weather Events and Canadian Geography",
        "Technology and Systems Engineering",
        "Linguistics and Terminology",
        "British Railway System and Colonial Influence",
        "Web Design and Formatting",
        "Royalty and Broadcasting",
        "Computer Science and Software",
        "Politics and Governance",
        "Sports and Olympic Competitions",
        "International Relations and Treaties",
        "Wrestling and Championships",
        "World Wars and Military History",
        "Museums and Recognition",
        "Midwestern U.S. Cities and Sports",
        "Wikipedia Articles and Discussion",
        "Education and Linguistics",
        "Companies and Media Businesses"
    ]

    cleandoc = clean(doc)
    doc_bow = ldamodel.id2word.doc2bow(cleandoc)
    doc_topics = ldamodel.get_document_topics(doc_bow, minimum_probability=0.05)

    if doc_topics:
        doc_topics.sort(key=itemgetter(1), reverse=True)
        best_topic_id = doc_topics[0][0]
        theme = topics[best_topic_id]

        if theme == "unknown" and len(doc_topics) > 1:
            best_topic_id = doc_topics[1][0]
            theme = topics[best_topic_id]
    else:
        best_topic_id = -1
        theme = "unknown"

    print(f"LDA Predicted Topic ID: {best_topic_id}, Theme: {theme}")
    return theme


def get_related_documents(term, top, corpus):
    print("-------------------",top," top articles related to ",term,"-----------------------")
    clean_docs = [clean(doc) for doc in corpus]
    related_docid = []
    test_term = [ldamodel.id2word.doc2bow(doc) for doc in clean_docs]
    doc_topics = ldamodel.get_document_topics(test_term, minimum_probability=0.20)        
    term_topics =  ldamodel.get_term_topics(term, minimum_probability=0.000001)
    for k,topics in enumerate(doc_topics):
        if topics:
            topics.sort(key = itemgetter(1), reverse=True)
            if topics[0][0] == term_topics[0][0]:
                related_docid.append((k,topics[0][1]))
    
    related_docid.sort(key = itemgetter(1), reverse=True)
    for j,doc_id in enumerate(related_docid):
        print(docs_test[doc_id[0]],"\n",doc_id[1],"\n")
        if j == (top-1):
            break


def cluster_similar_documents(corpus, dirname):
    clean_docs = [clean(doc) for doc in corpus]
    test_term = [ldamodel.id2word.doc2bow(doc) for doc in clean_docs]
    doc_topics = ldamodel.get_document_topics(test_term, minimum_probability=0.20)    
    for k,topics in enumerate(doc_topics):        
        if topics:
            topics.sort(key = itemgetter(1), reverse=True)
            dir_name = dirname + "/" + str(topics[0][0])           
            file_name = dir_name + "/" + str(k) + ".txt"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)    
            fp = open(file_name,"w")
            fp.write(docs_test[k] + "\n\n" + str(topics[0][1]) )
            fp.close()        
        else:           
            if not os.path.exists(dirname + "/unknown"):
                os.makedirs(dirname + "/unknown")  
            file_name = dirname + "/unknown/" + str(k) + ".txt"
            fp = open(file_name,"w")
            fp.write(docs_test[k]) 

docs_fp = open(os.path.join(current_path, "docs_wiki.pkl"), 'rb')
docs_all = pickle.load(docs_fp)
docs_test = docs_all[60000:]


if __name__ == "__main__" :
    # get_related_documents("music",5,docs_test)
    # cluster_similar_documents(docs_test,"root")
    article = "Music is the arrangement of sound to create some combination of form, harmony, melody, rhythm, or otherwise expressive content. Music is generally agreed to be a cultural universal that is present in all human societies. Definitions of music vary widely in substance and approach."
    print(article, "\n")

    print("Theme -> ",get_theme(article))