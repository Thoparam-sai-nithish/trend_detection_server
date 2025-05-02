#STEPS TO RUN THE MODEL
----------------------
1. Make sure you install the pyhton version 3.10.10 (Only this version is supported. use this version of python interpretor in vs code envronment)
2. Create a directory "privateKeys" and put your Google Cloud Service Account API key in it with name "content-based-trend-detection-key.json"
3. Go to the releases tab and download the binary file with name "bert_topic_model_custom_labels" and put it in the directory "bert_topic_modelling"
4. Run the below python script :
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
5. execute command in terminal : pip install -r requirements.txt
6. execute command in terminal : uvicorn server:app --reload --port 8000 
