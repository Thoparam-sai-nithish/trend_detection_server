import os
from deep_translator import GoogleTranslator
# from translate import Translator

translator = GoogleTranslator(source='auto', target='en')  # Auto-detect source language
# translator = Translator(from_lang="autodetect", to_lang="en")

def chunk_text(text, max_chars=450):
    chunks = []
    current_chunk = ""

    words = text.split(" ")

    for word in words:
        # Check if adding the next word exceeds the limit
        if len(current_chunk) + len(word) + 1 > max_chars:  # +1 for space
            chunks.append(current_chunk)
            current_chunk = ""

        # Add word to the current chunk
        if current_chunk:
            current_chunk += " "
        current_chunk += word

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def translate_text_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            with open(input_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            translated_text = ""

            try:
                chunks = chunk_text(text, max_chars=450)
                for chunk in chunks:
                    translated_text += translator.translate(chunk) + " "

            except Exception as e:
                print(f"Error translating {filename}: {e}")
                translated_text = text  # Fallback to original text
            
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(translated_text)
            
            print(f'🔤  Translated {filename} and saved to {output_path}')
    
    print('Translation completed!')


if __name__ == "__main__":
    # Define input and output folder paths
    input_folder = "textFiles"
    output_folder = "translatedFiles"

    # Run the translation function
    translate_text_files(input_folder, output_folder)
