import os
from utils import rmRecur
import librosa
import noisereduce as nr
from spleeter.separator import Separator
import soundfile as sf
from pydub import AudioSegment 
  

def cancelNoise(input_dir, file_name, output_dir, boost_factor = 2):
    file_path = os.path.join(input_dir, file_name)

    # Load original audio
    y, sr = librosa.load(file_path, sr=None)

    # Apply noise reduction
    noise_reduced_audio = nr.reduce_noise(y=y, sr=sr, 
                                          stationary=False,
                                          prop_decrease=0.7,
                                          n_fft=4096,
                                          thresh_n_mult_nonstationary=2)
    
    # amplify audio
    amplified_audio = noise_reduced_audio * boost_factor

    # Prevent clipping (ensure values are between -1 and 1)
    result_audio = amplified_audio / max(abs(amplified_audio).max(), 1)

    # Save to noiseCancelled directory
    output_path = os.path.join(output_dir, file_name)
    sf.write(output_path, result_audio, sr)

def separateVocals(input_dir, file_name, output_dir, separator):
    input_file_path = os.path.join(input_dir, file_name)

    # Initialize Spleeter with 2 stems (vocals + accompaniment) 
    # i.e, two files will be created in the specified output_directory/file_name/
    separator.separate_to_file(input_file_path, output_dir)


def batchNoiseCanceller(input_dir, output_dir):
    chunk_names = os.listdir(input_dir)
    for chunk_name in chunk_names:
        cancelNoise(input_dir, chunk_name, output_dir)

def batchVocalsSeparator(input_dir, output_dir, separator):
    chunk_names = os.listdir(input_dir)
    for chunk in chunk_names:
        separateVocals(input_dir, chunk, output_dir, separator)

def splitAudioFile(input_dir, file_name, output_dir, chunk_length = 30):
    input_file_path = os.path.join(input_dir, file_name)

    try:
        audio = AudioSegment.from_file(input_file_path)
    except Exception as e:
        print(f"❌ Error loading file {input_file_path}: {e}")
        return
    
    chunk_length_ms = chunk_length * 1000
    total_length_ms = len(audio)

    # Base filename (without extension) for saving chunks
    base_filename = os.path.splitext(file_name)[0]

    num_chunks = (total_length_ms + chunk_length_ms - 1) // chunk_length_ms
    
    for i in range(num_chunks):
        start_ms = i * chunk_length_ms
        end_ms = min(start_ms + chunk_length_ms, total_length_ms)
        chunk_audio = audio[start_ms:end_ms]
        
        # Define output filename for the chunk
        output_filename = f"{base_filename}_chunk_{i+1}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        # Export the chunk as a WAV file
        try:
            chunk_audio.export(output_path, format="wav")
            print(f"✅ Saved chunk {i+1} to {output_path}")
        except Exception as e:
            print(f"❌ Error exporting chunk {i+1}: {e}")


def preprocess(raw_audio_dir, spleetered_audio_dir, file_name, separator):
    file_path = os.path.join(raw_audio_dir, file_name)

    if not os.path.isfile(file_path):
        print(f"❌ {file_path} is not a file")
        return
    print(f"🔁 Preprocessing Audio {file_path}")


    # play audio
    # play(file_path)

    audio_chunks_dir = "audioChunks"
    noise_cancelled_dir = "noiseCancelled"
    preprocessed_dir = "preprocessedAudio"

    # Dividing into chunks
    os.makedirs(audio_chunks_dir, exist_ok=True)
    splitAudioFile(raw_audio_dir, file_name, audio_chunks_dir)

    # reduce noise
    os.makedirs(noise_cancelled_dir, exist_ok=True)
    print(f"🔇 Cancelling Noise From {file_path}")
    batchNoiseCanceller(audio_chunks_dir, noise_cancelled_dir)

    # remove music
    os.makedirs(spleetered_audio_dir, exist_ok=True)
    print(f"🎺 Seperatnig Vocals From {file_path}")
    batchVocalsSeparator(noise_cancelled_dir, spleetered_audio_dir, separator)

    rmRecur(audio_chunks_dir)
    rmRecur(noise_cancelled_dir)
    # aggregating chunks

    # save
    print(f"✅ successfully preprocessed {file_path}")