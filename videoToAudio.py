import os
import subprocess

def convertVideoToWav(video_dir, audio_dir):
    VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'webm', 'wmv', 'mpeg', 'mpg', '3gp', 'm4v'}
    os.makedirs(audio_dir, exist_ok=True)
    
    video_files = os.listdir(video_dir)
    for file_name in video_files:
        file_path = os.path.join(video_dir, file_name)

        # Check extension
        ext = file_name.split('.')[-1].lower()
        if ext not in VIDEO_EXTENSIONS:
            continue
            
        # Create output filename
        base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(audio_dir, f"{base_name}.wav")

        # FFmpeg command to extract audio
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-i', file_path,
            '-vn',  # Disable video recording
            '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
            '-ar', '44100',  # Sample rate
            '-ac', '2',  # Stereo audio
            '-loglevel', 'error',  # Only show errors
            output_path
        ]

        try:
            # Run FFmpeg command
            subprocess.run(cmd, check=True, 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE)
            print(f"⏭️  Converted: {file_name} -> {base_name}.wav")
        
        except subprocess.CalledProcessError as e:
            print(f"❌Error converting {file_name}: {e.stderr.decode()}")

        except FileNotFoundError:
            print("🫗FFmpeg not found! Please install FFmpeg and add it to PATH")
            break