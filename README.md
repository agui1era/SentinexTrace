# SentinexFace

SentinexFace is an advanced, real-time facial recognition and tracking system designed for multi-camera environments. It leverages computer vision, deep learning embeddings, and a reactive frontend to monitor RTSP streams, detect human subjects, and uniquely identify them over time across multiple camera feeds.

## Features

- **Multi-Camera Support:** Connect and monitor multiple RTSP streams simultaneously.
- **Real-Time Detection:** Uses OpenCV and YOLO/Haar integrations for rapid human detection.
- **Identity Tracking (CLIP + ChromaDB):** Uses OpenAI's CLIP model to generate facial embeddings and ChromaDB for high-dimensional vector search to group detections of the same person.
- **Automatic Tagging:** Automatically clusters unknown individuals. Users can retroactively tag an identity via the dashboard.
- **Historical Traceability:** Stores all detections and captures in MongoDB for querying full historical traces and analytics.
- **Modern React Dashboard:** A responsive, glassmorphism UI built with React and Vite for monitoring cameras, managing identities, and reviewing historical captures.

## Architecture

1. **Frontend (React/Vite):** Located in `src/`. Connects to the backend via REST APIs.
2. **Backend API (Python/FastAPI):** Located in `backend/`. Handles database interactions, ChromaDB embedding lookups, and stream analysis.
3. **RTSP Bridge (Node.js):** Located in `server/`. Bridges RTSP streams to WebRTC or MJPEG formats suitable for modern browsers.

## Prerequisites

- **Node.js** (v18+)
- **Python** (3.9+)
- **MongoDB** (Running locally on port 27017)

## Installation & Startup

A unified startup script is provided for easy deployment across machines. It will automatically check for dependencies, install them if missing, kill any ghost processes, and start all three microservices.

```bash
chmod +x start-all.sh
./start-all.sh
```

### Manual Configuration

Environment variables are managed in the `.env` file. To add new cameras, use the `VITE_RTSP_URL_CAM` prefix:

```env
VITE_RTSP_URL_CAM1=rtsp://user:password@ip:554/stream1
VITE_RTSP_URL_CAM2=rtsp://user:password@ip:554/stream2
```

## Tech Stack

- **UI:** React, Vite, CSS Modules, Lucide Icons
- **Backend:** Python, FastAPI, Uvicorn, OpenCV (cv2)
- **AI/ML:** HuggingFace Transformers, ChromaDB, CLIP
- **Database:** MongoDB
- **Video Processing:** RTSP Protocol, MJPEG Streaming

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
