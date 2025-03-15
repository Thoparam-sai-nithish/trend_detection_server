from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from main import main
import os, uvicorn, shutil 
 
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to FastAPI Server"}

@app.post("/upload-videos/")
async def upload_videos(files: List[UploadFile] = File(...)):
    saved_files = []
    VIDEOS_DIR = "videos"
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    print("Request Recieved!")
    for file in files: 
        video_path = os.path.join(VIDEOS_DIR, file.filename)

        # Save each video to the 'videos' directory
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_files.append(video_path)
 
    main()
    

    return {"message": "Videos uploaded successfully", "saved_files": saved_files}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

