# Video Object Tracker

A real-time video processing system that automatically detects and tracks objects across video chunks using YOLOv8. Perfect for surveillance, monitoring, and video analysis applications.

## Quick Start

### 1. Installation

**Prerequisites:**
- Python 3.7 or higher
- Git

**Setup:**
```bash
# Clone or download this project
cd Server

# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Server

**Start the API server:**
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

You'll see:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Optional: Expose publicly with ngrok:**
```bash
ngrok http 8000
```

This gives you a public URL to access your server from anywhere.

## Using the Server

### Upload Videos
```bash
curl -F "video=@your_video.mp4" http://localhost:8000/upload
```

Or use the FastAPI Swagger UI at `http://localhost:8000/docs`

### Health Check
```bash
curl http://localhost:8000/health
```

## How It Works

1. **Upload** - Place video files in the `uploads/` folder or use the `/upload` endpoint
2. **Process** - The system automatically detects objects using YOLOv8 AI model
3. **Track** - Objects are tracked across video frames and linked across sequential chunks
4. **Store** - Results are saved in an SQLite database (`objects.db`)
5. **Query** - Inspect tracked objects using the database

## File Naming Convention

Video filenames should include timestamps in this format:
```
video_YYYYMMDDThhmmssZ_[number].mp4
```

Example: `video_20260103T112511Z_100.mp4`

This ensures videos are processed in chronological order for accurate track linking.

## Folder Structure

```
uploads/                      - Drop video files here for processing
processed/
├── {video_name}/            - One folder per video
│   ├── labels/              - Detection labels per frame
│   │   ├── {video}_1.txt
│   │   ├── {video}_2.txt
│   │   └── ...
│   └── {video}.avi          - Processed video with detections
├── {another_video}/
│   ├── labels/
│   └── ...
models/
└── best.pt                  - YOLOv8 model weights
objects.db                   - SQLite database with tracked objects
```

## Advanced Usage

**Inspect tracked objects:**
```bash
python inspect_db.py --db objects.db
```

**Process videos manually:**
```bash
python inference.py uploads --project processed
```

## Troubleshooting

- **"Module not found" error** - Make sure virtual environment is activated and dependencies installed
- **Port 8000 already in use** - Change port: `--port 8001`
- **Videos not processing** - Check filenames include proper timestamp format
- **Database locked** - Ensure only one server instance is running

## Support

For issues or questions, check the configuration in `app.py` or adjust tracking thresholds in `tracker.py`
