import os 
import io
import subprocess
from google.oauth2 import service_account
from google.cloud import speech_v1p1beta1 as speech
import whisper
openai_model = whisper.load_model("tiny")

def detect_language(input_file_path):
    LANGUAGE_CODE_MAPPING = {
        'as': 'as-IN',  # Assamese
        'bn': 'bn-IN',  # Bengali
        'en': 'en-US',  # English (India)
        'gu': 'gu-IN',  # Gujarati
        'hi': 'hi-IN',  # Hindi
        'kn': 'kn-IN',  # Kannada
        'ml': 'ml-IN',  # Malayalam
        'mr': 'mr-IN',  # Marathi
        'or': 'or-IN',  # Odia
        'pa': 'pa-IN',  # Punjabi
        'ta': 'ta-IN',  # Tamil
        'te': 'te-IN',  # Telugu
    }
    
    result = openai_model.transcribe(input_file_path)
    detected_language = result.get('language')

    if detected_language :
        google_language_code = LANGUAGE_CODE_MAPPING.get(detected_language)
        if google_language_code:
            return google_language_code

    return "en-IN"

def convert_to_text(input_file_path, output_file_path, language_code = "en-US", stt_model="STT"):
    if not os.path.exists(input_file_path):
        print(f"⚠️  Skipping: {input_file_path} (File not found)")
        return
    

    try:
        print(f"⛏️  Extracting text from {input_file_path}...")

# # # # # # # # # # #  SPEECH TO TEXT USING GOOGLE CLOUD API # # # # # # # # # # #

        # Convert to mono using FFmpeg
        mono_audio = "mono_audio.wav"
        subprocess.run([
            "ffmpeg", "-i", input_file_path, "-ac", "1", "-ar", "44100", "-f", "wav", mono_audio
        ], check=True, stderr=subprocess.DEVNULL) 

        key_file_path = os.path.join('privateKeys', 'content-based-trend-detection-key.json')
        credentials = service_account.Credentials.from_service_account_file(key_file_path)
        client = speech.SpeechClient(credentials=credentials)


        with open(mono_audio, "rb") as audio_file: content = audio_file.read()
        
        audio = speech.RecognitionAudio(content=content)
        
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz = 44100,
            enable_automatic_punctuation=True,
            language_code = language_code,
        )

        response = client.recognize(config=config, audio=audio)
        
        if not response.results:
            print(f"⚠️ No speech detected in {input_file_path}")
            return
        
        text = " ".join(result.alternatives[0].transcript for result in response.results).strip()

        with open(output_file_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
            print(f"✅ Text appended to {output_file_path}")
        
    except Exception as e:
        print(f"❌ Error processing {input_file_path}: {e}")
    
    finally:
         if os.path.exists(mono_audio):
             os.remove(mono_audio)

# # # # # # # # # # #  SPEECH TO TEXT USING GOOGLE STT API # # # # # # # # # # #

    #     recognizer = sr.Recognizer()
    #     with sr.AudioFile(input_file_path) as audio_file:
    #         audio_data = recognizer.record(audio_file)
    #         # Auto-detect language using Google Speech API
    #         text = recognizer.recognize_google(audio_data)
    #         print(f"Text Extracted is : \n {text}")
    #         # Append transcribed text to the output file
    #         with open(output_file_path, "a", encoding="utf-8") as f:
    #             f.write(text + "\n")
    #             print(f"✅  Text appended to {output_file_path}")

    # except sr.UnknownValueError:
    #     print(f"⚠️  Could not understand speech in {input_file_path}")
    # except sr.RequestError as e:
    #     print(f"❌  Error connecting to Google Speech API: {e}")
    # except Exception as e:
    #     print(f"❌  Error processing {input_file_path}: {e}")


# # # # # # # # # # #  SPEECH TO TEXT USING OPEN-AI-WHISPER MODEL # # # # # # # # # # #
        
    #     # stt_model = WhisperModel("large", device="cpu", compute_type="int8")

    #     segments, info = stt_model.transcribe(input_file_path)
        
    #     text = " ".join(segment.text for segment in segments).strip()
    #     # print(f"Text : \n{text}")

    #     if not text:
    #         print(f"⚠️  No speech detected in {input_file_path}")
    #         return

    #     with open(output_file_path, "a", encoding="utf-8") as f:
    #         f.write(text + "\n")
    #         print(f"✅  Text appended to {output_file_path}")

    # except Exception as e:
    #     print(f"❌  Error processing {input_file_path}: {e}")


def batchSpeechToText(input_dir, output_dir, base_file_name, stt_model):
    print(f"🔠 Converting {base_file_name} chunks into text")

    if not os.path.exists(input_dir):
        print(f"❌  Direcotry '{input_dir}' does not exists")
        return
    
    chunks_dirs_list = os.listdir(input_dir)
    if len(chunks_dirs_list) == 0:
        print("❌  No Chunks Found")
        return
    
    print(f"📂 Spleetered chunks found: {chunks_dirs_list}")
    
    os.makedirs(output_dir, exist_ok=True)
    output_file_name = os.path.splitext(base_file_name)[0]+'.txt'
    output_file_path = os.path.join(output_dir, output_file_name)
 
    with open(output_file_path, "w", encoding="utf-8") as f:
        print(f"📝 Created a empty file at {output_file_path}")

    language_code =  detect_language(os.path.join(input_dir, chunks_dirs_list[0], "vocals.wav"))
    print(f"Detected Language : {language_code}")

    for chunk_dir in chunks_dirs_list:
        input_file_path = os.path.join(input_dir, chunk_dir,"vocals.wav")
        convert_to_text(input_file_path,  output_file_path, language_code, stt_model)


if __name__ == "__main__":
    input_file_path = os.path.join('rawAudio', 'tamil-content.wav')
    output_file_path = os.path.join('textFiles', 'tamilText.txt')
    with open(output_file_path, "w", encoding="utf-8") as f:
        print(f"📝 Created a empty file at {output_file_path}")

    language_code =  detect_language(input_file_path)
    print(f"Detected Languge : {language_code}")
    convert_to_text(input_file_path, output_file_path, language_code, "STT")