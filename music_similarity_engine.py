import librosa
import numpy as np
import sounddevice as sd
from scipy.spatial.distance import cosine
from collections import deque
import threading
import time
from typing import List, Dict, Tuple
import pickle
import os

class AudioFingerprinter:
    """Extract audio features for similarity comparison"""
    
    def __init__(self, sr=22050, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_features(self, audio_segment: np.ndarray) -> Dict:
        """Extract comprehensive audio features"""
        
        # Ensure audio is mono and float32
        if len(audio_segment.shape) > 1:
            audio_segment = librosa.to_mono(audio_segment.T)
        
        features = {}
        
        try:
            # 1. MFCCs (Mel-frequency cepstral coefficients)
            mfcc = librosa.feature.mfcc(y=audio_segment, sr=self.sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            
            # 2. Chroma features (pitch class profiles)
            chroma = librosa.feature.chroma_stft(y=audio_segment, sr=self.sr)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            
            # 3. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_segment, sr=self.sr)
            features['spectral_centroid'] = np.mean(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=self.sr)
            features['spectral_rolloff'] = np.mean(spectral_rolloff)
            
            # 4. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_segment)
            features['zcr'] = np.mean(zcr)
            
            # 5. Tempo and beat features
            tempo, beats = librosa.beat.beat_track(y=audio_segment, sr=self.sr)
            features['tempo'] = tempo
            
            # 6. RMS Energy
            rms = librosa.feature.rms(y=audio_segment)
            features['rms'] = np.mean(rms)
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
            
        return features
    
    def features_to_vector(self, features: Dict) -> np.ndarray:
        """Convert feature dict to numerical vector"""
        if features is None:
            return np.zeros(32)  # Return zero vector on error
            
        vector = []
        
        # Flatten arrays and add scalars
        mfcc_mean = features.get('mfcc_mean', np.zeros(13))
        if hasattr(mfcc_mean, 'flatten'):
            vector.extend(mfcc_mean.flatten())
        else:
            vector.extend([mfcc_mean] if np.isscalar(mfcc_mean) else mfcc_mean)
            
        mfcc_std = features.get('mfcc_std', np.zeros(13))
        if hasattr(mfcc_std, 'flatten'):
            vector.extend(mfcc_std.flatten())
        else:
            vector.extend([mfcc_std] if np.isscalar(mfcc_std) else mfcc_std)
            
        chroma_mean = features.get('chroma_mean', np.zeros(12))
        if hasattr(chroma_mean, 'flatten'):
            vector.extend(chroma_mean.flatten())
        else:
            vector.extend([chroma_mean] if np.isscalar(chroma_mean) else chroma_mean)
        
        # Add scalar values
        vector.append(float(features.get('spectral_centroid', 0)))
        vector.append(float(features.get('spectral_rolloff', 0)))
        vector.append(float(features.get('zcr', 0)))
        vector.append(float(features.get('tempo', 0)))
        vector.append(float(features.get('rms', 0)))
        
        return np.array(vector, dtype=np.float32)


class MusicDatabase:
    """Store and search existing music fingerprints"""
    
    def __init__(self, db_path="music_db.pkl"):
        self.db_path = db_path
        self.fingerprints = {}  # {song_id: {'vector': np.array, 'metadata': dict}}
        self.load_database()
    
    def add_song(self, song_id: str, audio_file: str, metadata: Dict = None):
        """Add a song to the database"""
        try:
            audio, sr = librosa.load(audio_file, sr=22050)
            fingerprinter = AudioFingerprinter()
            
            # Process song in chunks to get multiple fingerprints
            chunk_size = sr * 5  # 5-second chunks
            vectors = []
            
            for i in range(0, len(audio) - chunk_size, chunk_size // 2):
                chunk = audio[i:i + chunk_size]
                features = fingerprinter.extract_features(chunk)
                if features:
                    vector = fingerprinter.features_to_vector(features)
                    vectors.append(vector)
            
            if vectors:
                # Use mean vector as song representation
                song_vector = np.mean(vectors, axis=0)
                self.fingerprints[song_id] = {
                    'vector': song_vector,
                    'metadata': metadata or {},
                    'chunks': vectors  # Store individual chunks for detailed comparison
                }
                self.save_database()
                print(f"Added song: {song_id}")
                
        except Exception as e:
            print(f"Error adding song {song_id}: {e}")
    
    def find_similar(self, query_vector: np.ndarray, threshold: float = 0.8) -> List[Tuple[str, float]]:
        """Find similar songs based on cosine similarity"""
        similarities = []
        
        fingerprints_data = self.fingerprints
        
        # Handle nested fingerprints structure
        if "fingerprints" in fingerprints_data:
            fingerprints_data = fingerprints_data["fingerprints"]
        
        for song_id, data in fingerprints_data.items():
            if isinstance(data, dict) and 'vector' in data:
                stored_vector = data['vector']
                
                # Calculate cosine similarity (1 - cosine_distance)
                similarity = 1 - cosine(query_vector, stored_vector)
                
                if similarity > threshold:
                    similarities.append((song_id, similarity, data.get('metadata', {})))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def save_database(self):
        """Save database to disk"""
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.fingerprints, f)
    
    def load_database(self):
        """Load database from disk"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    self.fingerprints = pickle.load(f)
                print(f"Loaded {len(self.fingerprints)} songs from database")
            except Exception as e:
                print(f"Error loading database: {e}")


class RealTimeSimilarityDetector:
    """Real-time audio capture and similarity detection"""
    
    def __init__(self, database: MusicDatabase, sample_rate=22050):
        self.database = database
        self.sample_rate = sample_rate
        self.fingerprinter = AudioFingerprinter(sr=sample_rate)
        
        # Audio buffer for real-time processing
        self.buffer_size = sample_rate * 5  # 5 seconds
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.is_recording = False
        
        # Similarity detection settings
        self.similarity_threshold = 0.75
        self.check_interval = 2.0  # Check every 2 seconds
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for real-time audio capture"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Add new audio data to buffer
        audio_data = indata[:, 0] if len(indata.shape) > 1 else indata
        self.audio_buffer.extend(audio_data)
    
    def start_monitoring(self, similarity_callback=None):
        """Start real-time similarity monitoring"""
        self.is_recording = True
        
        def detection_loop():
            while self.is_recording:
                if len(self.audio_buffer) >= self.buffer_size:
                    # Convert buffer to numpy array
                    audio_segment = np.array(list(self.audio_buffer))
                    
                    # Extract features and check similarity
                    features = self.fingerprinter.extract_features(audio_segment)
                    if features:
                        query_vector = self.fingerprinter.features_to_vector(features)
                        similar_songs = self.database.find_similar(
                            query_vector, 
                            threshold=self.similarity_threshold
                        )
                        
                        if similar_songs and similarity_callback:
                            similarity_callback(similar_songs)
                
                time.sleep(self.check_interval)
        
        # Start detection thread
        detection_thread = threading.Thread(target=detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        
        # Start audio stream
        try:
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=1024
            ):
                print("Started real-time monitoring. Press Ctrl+C to stop.")
                while self.is_recording:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_monitoring()
        except Exception as e:
            print(f"Error in audio stream: {e}")
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_recording = False
        print("Stopped monitoring")


if __name__ == "__main__":
    # Example usage
    db = MusicDatabase()
    
    # Add some songs to the database
    # db.add_song("song1", "path/to/song1.mp3", {"artist": "Artist 1", "genre": "Rock"})
    
    # Start real-time detection
    detector = RealTimeSimilarityDetector(db)
    
    def on_similarity_found(similar_songs):
        print("Similarity detected!")
        for song_id, similarity, metadata in similar_songs:
            print(f"  {song_id}: {similarity:.2f} similarity")
            if metadata:
                print(f"    Metadata: {metadata}")
    
    detector.start_monitoring(similarity_callback=on_similarity_found)