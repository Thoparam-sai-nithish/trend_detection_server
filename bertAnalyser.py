import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bert_topic_modelling.topic_model import get_theme

def bert_analyser(input_dir):
    themes = {} 

    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)

        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                text_content = file.read()

            theme = get_theme(text_content)

            themes[file_name] = theme

    return themes

if __name__ == "__main__":
    result = bert_analyser('translatedFiles')
    for key in result.keys():
        print(f"{key} : {result[key]}")