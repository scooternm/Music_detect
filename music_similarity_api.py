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
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel

# Import our similarity detection classes
try:
    from music_similarity_engine import AudioFingerprinter, MusicDatabase, RealTimeSimilarityDetector
except ImportError as e:
    logging.error(f"Failed to import music_similarity_engine: {e}")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Music Similarity Detection API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances with error handling
try:
    music_db = MusicDatabase()
    fingerprinter = AudioFingerprinter()
    logger.info("Successfully initialized music database and fingerprinter")
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    raise

# Pydantic models for API
class SongMetadata(BaseModel):
    artist: str = ""
    title: str = ""
    genre: str = ""
    year: Optional[int] = None

class SimilarityResult(BaseModel):
    song_id: str
    similarity_score: float
    metadata: Dict

class ConnectionManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        logger.info("ConnectionManager initialized")
    
    async def connect(self, websocket: WebSocket):
        try:
            await websocket.accept()
            self.active_connections.append(websocket)
            logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            raise
    
    def disconnect(self, websocket: WebSocket):
        try:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket: {e}")
    
    async def broadcast(self, message: dict):
        disconnected_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send message to connection: {e}")
                disconnected_connections.append(connection)
        
        # Remove dead connections
        for connection in disconnected_connections:
            self.disconnect(connection)

manager = ConnectionManager()

@app.get("/")
async def root():
    return {"message": "Music Similarity Detection API", "version": "1.0.0"}

@app.post("/upload-song")
async def upload_song(
    file: UploadFile = File(...),
    song_id: Optional[str] = None,
    metadata: str = "{}"
):
    """Upload a song to the database"""
    
    logger.info(f"Uploading song: {file.filename}")
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    
    if not song_id:
        song_id = os.path.splitext(file.filename)[0]
    
    temp_file_path = None
    try:
        # Parse metadata with validation
        try:
            song_metadata = json.loads(metadata)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid metadata JSON: {e}")
            raise HTTPException(status_code=400, detail="Invalid metadata JSON format")
        
        # Validate file size (e.g., max 50MB)
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large (max 50MB)")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"Saved temporary file: {temp_file_path}")
        
        # Add to database with error handling
        try:
            music_db.add_song(song_id, temp_file_path, song_metadata)
            logger.info(f"Successfully added song '{song_id}' to database")
        except Exception as e:
            logger.error(f"Failed to add song to database: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
        return {
            "message": f"Song '{song_id}' uploaded successfully",
            "song_id": song_id,
            "metadata": song_metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error uploading song: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing song: {str(e)}")
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")

@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    """Analyze uploaded audio and find similarities"""
    
    logger.info(f"Analyzing audio file: {file.filename}")
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    
    temp_file_path = None
    try:
        # Read and validate file
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Load and process audio
        try:
            audio, sr = librosa.load(temp_file_path, sr=22050)
            logger.info(f"Loaded audio: duration={len(audio)/sr:.2f}s, sr={sr}")
        except Exception as e:
            logger.error(f"Failed to load audio with librosa: {e}")
            raise HTTPException(status_code=400, detail="Invalid audio file or format not supported")
        
        if len(audio) == 0:
            raise HTTPException(status_code=400, detail="Audio file contains no data")
        
        # Extract features
        try:
            features = fingerprinter.extract_features(audio)
            if not features:
                raise HTTPException(status_code=400, detail="Could not extract audio features")
            logger.info(f"Extracted {len(features)} features from audio")
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Feature extraction error: {str(e)}")
        
        try:
            query_vector = fingerprinter.features_to_vector(features)
            logger.info(f"Created query vector with shape: {np.array(query_vector).shape}")
        except Exception as e:
            logger.error(f"Vector conversion failed: {e}")
            raise HTTPException(status_code=500, detail=f"Vector conversion error: {str(e)}")
        
        # Find similarities
        try:
            similar_songs = music_db.find_similar(query_vector, threshold=0.7)
            logger.info(f"Found {len(similar_songs)} similar songs")
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise HTTPException(status_code=500, detail=f"Similarity search error: {str(e)}")
        
        # Format results
        results = []
        for song_id, similarity, metadata in similar_songs:
            try:
                results.append(SimilarityResult(
                    song_id=song_id,
                    similarity_score=float(similarity),
                    metadata=metadata if isinstance(metadata, dict) else {}
                ))
            except Exception as e:
                logger.warning(f"Failed to format result for song {song_id}: {e}")
        
        return {
            "similar_songs": [result.dict() for result in results],
            "total_matches": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error analyzing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing audio: {str(e)}")
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")

@app.get("/database/songs")
async def get_database_songs():
    """Get all songs in the database"""
    
    try:
        songs = []
        fingerprints_data = music_db.fingerprints
        
        # Handle different fingerprints structure formats
        if isinstance(fingerprints_data, dict):
            if "fingerprints" in fingerprints_data:
                fingerprints_data = fingerprints_data["fingerprints"]
            
            for song_id, data in fingerprints_data.items():
                try:
                    if isinstance(data, dict):
                        songs.append({
                            "song_id": song_id,
                            "metadata": data.get("metadata", {}),
                            "features_count": len(data.get("chunks", [])) if data.get("chunks") else 0
                        })
                    else:
                        # Handle string entries (legacy format)
                        songs.append({
                            "song_id": song_id,
                            "metadata": {},
                            "features_count": 1
                        })
                except Exception as e:
                    logger.warning(f"Error processing song {song_id}: {e}")
        
        logger.info(f"Retrieved {len(songs)} songs from database")
        
        return {
            "songs": songs,
            "total_songs": len(songs)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving database songs: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.delete("/database/songs/{song_id}")
async def delete_song(song_id: str):
    """Delete a song from the database"""
    
    try:
        fingerprints_data = music_db.fingerprints
        
        # Handle nested fingerprints structure
        found = False
        if isinstance(fingerprints_data, dict):
            if "fingerprints" in fingerprints_data:
                if song_id in fingerprints_data["fingerprints"]:
                    del fingerprints_data["fingerprints"][song_id]
                    found = True
            else:
                if song_id in fingerprints_data:
                    del fingerprints_data[song_id]
                    found = True
        
        if not found:
            raise HTTPException(status_code=404, detail="Song not found")
        
        # Save database
        try:
            music_db.save_database()
            logger.info(f"Successfully deleted song '{song_id}'")
        except Exception as e:
            logger.error(f"Failed to save database after deletion: {e}")
            raise HTTPException(status_code=500, detail="Failed to save database")
        
        return {"message": f"Song '{song_id}' deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting song: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting song: {str(e)}")

@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time similarity detection"""
    
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive audio data from client
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue
            except Exception as e:
                logger.error(f"Error receiving WebSocket data: {e}")
                break
            
            if message.get("type") == "audio_chunk":
                try:
                    # Decode base64 audio data
                    audio_b64 = message.get("audio_data")
                    if not audio_b64:
                        await websocket.send_text(json.dumps({"type": "error", "message": "No audio data"}))
                        continue
                    
                    audio_bytes = base64.b64decode(audio_b64)
                    
                    # Convert to numpy array (assuming 16-bit PCM)
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    if len(audio_np) == 0:
                        continue
                    
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
                                        "metadata": metadata if isinstance(metadata, dict) else {}
                                    }
                                    for song_id, similarity, metadata in similar_songs[:5]
                                ]
                            }
                            await websocket.send_text(json.dumps(response))
                            
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}")
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
            
            elif message.get("type") == "ping":
                # Heartbeat
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected normally")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        db_size = len(music_db.fingerprints)
        if isinstance(music_db.fingerprints, dict) and "fingerprints" in music_db.fingerprints:
            db_size = len(music_db.fingerprints["fingerprints"])
        
        return {
            "status": "healthy",
            "database_songs": db_size,
            "active_connections": len(manager.active_connections),
            "librosa_version": librosa.__version__,
            "numpy_version": np.__version__
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Additional utility endpoints
@app.post("/batch-analyze")
async def batch_analyze(files: List[UploadFile] = File(...)):
    """Analyze multiple audio files for similarities"""
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Too many files (max 10)")
    
    results = []
    
    for file in files:
        temp_file_path = None
        try:
            # Validate file type
            if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                results.append({
                    "filename": file.filename,
                    "error": "Unsupported audio format"
                })
                continue
            
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
                            "similarity": float(similarity),
                            "metadata": metadata if isinstance(metadata, dict) else {}
                        }
                        for song_id, similarity, metadata in similar_songs
                    ]
                })
            else:
                results.append({
                    "filename": file.filename,
                    "error": "Could not extract features"
                })
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up {temp_file_path}: {e}")
    
    return {"results": results}

# Debug endpoint for development
@app.get("/debug/database")
async def debug_database():
    """Debug endpoint to inspect database structure"""
    try:
        return {
            "fingerprints_type": type(music_db.fingerprints).__name__,
            "fingerprints_keys": list(music_db.fingerprints.keys()) if isinstance(music_db.fingerprints, dict) else "Not a dict",
            "sample_entry": dict(list(music_db.fingerprints.items())[:1]) if isinstance(music_db.fingerprints, dict) and music_db.fingerprints else "Empty or not dict"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting Music Similarity Detection API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Debug endpoint: http://localhost:8000/debug/database")
    
    uvicorn.run(
        "music_similarity_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )