from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import json
import base64
import numpy as np
import librosa
from io import BytesIO
import tempfile
import os
from typing import List, Dict
from pydantic import BaseModel

# Import our similarity detection classes
from music_similarity_engine import AudioFingerprinter, MusicDatabase, RealTimeSimilarityDetector

app = FastAPI(title="Music Similarity Detection API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
music_db = MusicDatabase()
fingerprinter = AudioFingerprinter()

# Pydantic models for API
class SongMetadata(BaseModel):
    artist: str = ""
    title: str = ""
    genre: str = ""
    year: int = None

class SimilarityResult(BaseModel):
    song_id: str
    similarity_score: float
    metadata: Dict

class ConnectionManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                # Remove dead connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

@app.get("/")
async def root():
    return {"message": "Music Similarity Detection API", "version": "1.0.0"}

@app.post("/upload-song")
async def upload_song(
    file: UploadFile = File(...),
    song_id: str = None,
    metadata: str = "{}"
):
    """Upload a song to the database"""
    
    if not song_id:
        song_id = file.filename.split('.')[0]
    
    try:
        # Parse metadata
        song_metadata = json.loads(metadata)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Add to database
        music_db.add_song(song_id, temp_file_path, song_metadata)
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        return {
            "message": f"Song '{song_id}' uploaded successfully",
            "song_id": song_id,
            "metadata": song_metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing song: {str(e)}")

@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    """Analyze uploaded audio and find similarities"""
    
    try:
        # Read audio file
        content = await file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Load and process audio
        audio, sr = librosa.load(temp_file_path, sr=22050)
        
        # Extract features
        features = fingerprinter.extract_features(audio)
        if not features:
            raise HTTPException(status_code=400, detail="Could not extract audio features")
        
        query_vector = fingerprinter.features_to_vector(features)
        
        # Find similarities
        similar_songs = music_db.find_similar(query_vector, threshold=0.7)
        
        # Clean up
        os.unlink(temp_file_path)
        
        # Format results
        results = [
            SimilarityResult(
                song_id=song_id,
                similarity_score=similarity,
                metadata=metadata
            )
            for song_id, similarity, metadata in similar_songs
        ]
        
        return {
            "similar_songs": [result.dict() for result in results],
            "total_matches": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error analyzing audio: {str(e)}")

@app.get("/database/songs")
async def get_database_songs():
    """Get all songs in the database"""
    
    songs = []
    for song_id, data in music_db.fingerprints.items():
        songs.append({
            "song_id": song_id,
            "metadata": data.get("metadata", {}),
            "features_count": len(data.get("chunks", []))
        })
    
    return {
        "songs": songs,
        "total_songs": len(songs)
    }

@app.delete("/database/songs/{song_id}")
async def delete_song(song_id: str):
    """Delete a song from the database"""
    
    if song_id not in music_db.fingerprints:
        raise HTTPException(status_code=404, detail="Song not found")
    
    del music_db.fingerprints[song_id]
    music_db.save_database()
    
    return {"message": f"Song '{song_id}' deleted successfully"}

@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time similarity detection"""
    
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive audio data from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "audio_chunk":
                # Decode base64 audio data
                audio_b64 = message["audio_data"]
                audio_bytes = base64.b64decode(audio_b64)
                
                # Convert to numpy array (assuming 16-bit PCM)
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Extract features
                features = fingerprinter.extract_features(audio_np)
                if features:
                    query_vector = fingerprinter.features_to_vector(features)
                    similar_songs = music_db.find_similar(query_vector, threshold=0.75)
                    
                    if similar_songs:
                        # Send similarity results back
                        response = {
                            "type": "similarity_detected",
                            "matches": [
                                {
                                    "song_id": song_id,
                                    "similarity": float(similarity),
                                    "metadata": metadata
                                }
                                for song_id, similarity, metadata in similar_songs[:5]
                            ]
                        }
                        await websocket.send_text(json.dumps(response))
            
            elif message["type"] == "ping":
                # Heartbeat
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database_songs": len(music_db.fingerprints),
        "active_connections": len(manager.active_connections)
    }

# Additional utility endpoints
@app.post("/batch-analyze")
async def batch_analyze(files: List[UploadFile] = File(...)):
    """Analyze multiple audio files for similarities"""
    
    results = []
    
    for file in files:
        try:
            content = await file.read()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            audio, sr = librosa.load(temp_file_path, sr=22050)
            features = fingerprinter.extract_features(audio)
            
            if features:
                query_vector = fingerprinter.features_to_vector(features)
                similar_songs = music_db.find_similar(query_vector, threshold=0.7)
                
                results.append({
                    "filename": file.filename,
                    "matches": [
                        {
                            "song_id": song_id,
                            "similarity": similarity,
                            "metadata": metadata
                        }
                        for song_id, similarity, metadata in similar_songs
                    ]
                })
            
            os.unlink(temp_file_path)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}

if __name__ == "__main__":
    print("Starting Music Similarity Detection API...")
    print("API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        "music_similarity_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )