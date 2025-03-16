import pickle
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from operator import itemgetter
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

stop = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

current_path = os.path.dirname(__file__)
lda_fp = open(os.path.join(current_path, "lda_model_sym_wiki.pkl"), 'rb')
ldamodel = pickle.load(lda_fp)

def rem_ascii(s):
    return "".join([c for c in s if ord(c) < 128 ])

def clean_doc(doc):
    doc_ascii = rem_ascii(doc)
    stop_free = " ".join([i for i in doc_ascii.lower().split() if i not in stop])
    normalized = " ".join(lemma.lemmatize(word,'v') for word in stop_free.split())
    x = normalized.split()
    y = [s for s in x if len(s) > 2]
    return y

def get_theme(doc):
    topics = "politics usa literature geography chemistry london awards sports california monarchy biology census commerce health technology botany education government cinema wikipedia philosophy templates football history wrestling racing music travel television war mortality albums business canada cities culture entertainment france germany space nature crime astronomy mathematics gaming hurricane disaster law physics medicine".split()
    
    
    theme = ""
    cleandoc = clean_doc(doc)
    doc_bow = ldamodel.id2word.doc2bow(cleandoc)
    doc_topics = ldamodel.get_document_topics(doc_bow, minimum_probability=0.05)

    # Print the Topcis that the model is trained and identified 
    for idx, topic in ldamodel.print_topics(num_topics=50, num_words=10):
        print(f"Topic {idx}: {topic}")

    if doc_topics:
        doc_topics.sort(key = itemgetter(1), reverse=True)
        theme = topics[doc_topics[0][0]]
        if theme == "unknown":
            theme = topics[doc_topics[1][0]]
    else:
        theme = "unknown"

    return theme


def get_related_documents(term, top, corpus):
    print("-------------------",top," top articles related to ",term,"-----------------------")
    clean_docs = [clean_doc(doc) for doc in corpus]
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
    clean_docs = [clean_doc(doc) for doc in corpus]
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
    article = 'Gaming refers to the activity of playing video games on various platforms like computers, consoles, and mobile devices, encompassing a wide range of genres including action, adventure, strategy, and simulation. It has become a major form of entertainment for people of all ages, allowing players to immerse themselves in virtual worlds, complete challenges, and interact with other players online, often fostering a sense of community and providing opportunities for creative expression and problem-solving skills development. While leisure gaming is prevalent, the industry also includes competitive esports, where professional gamers compete in tournaments for significant prize pools.'
    print(article, "\n")

    print("Theme -> ",get_theme(article))