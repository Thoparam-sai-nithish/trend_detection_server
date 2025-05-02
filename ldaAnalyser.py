import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lda_topic_modelling.wiki_topic_cluster import get_theme

def lda_analyser(input_dir):
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
    print(lda_analyser('translatedFiles'))