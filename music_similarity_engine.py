import librosa
import numpy as np
import sounddevice as sd
from scipy.spatial.distance import cosine
from collections import deque
import threading
import time
from typing import List, Dict, Tuple, Optional, Callable
import pickle
import os
import warnings

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=UserWarning)

class AudioFingerprinter:
    """Extract audio features for similarity comparison"""
    
    def __init__(self, sr=22050, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_features(self, audio_segment: np.ndarray) -> Optional[Dict]:
        """Extract comprehensive audio features"""
        
        # Input validation
        if audio_segment is None or len(audio_segment) == 0:
            print("Warning: Empty audio segment")
            return None
            
        # Ensure audio is mono and float32
        if len(audio_segment.shape) > 1:
            audio_segment = librosa.to_mono(audio_segment.T)
        
        # Ensure minimum length for feature extraction
        min_length = self.hop_length * 4  # Minimum 4 frames
        if len(audio_segment) < min_length:
            # Pad with zeros if too short
            audio_segment = np.pad(audio_segment, (0, min_length - len(audio_segment)))
        
        features = {}
        
        try:
            # 1. MFCCs (Mel-frequency cepstral coefficients)
            mfcc = librosa.feature.mfcc(
                y=audio_segment, 
                sr=self.sr, 
                n_mfcc=13,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            
            # 2. Chroma features (pitch class profiles)
            chroma = librosa.feature.chroma_stft(
                y=audio_segment, 
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            features['chroma_mean'] = np.mean(chroma, axis=1)
            
            # 3. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_segment, 
                sr=self.sr,
                hop_length=self.hop_length
            )
            features['spectral_centroid'] = np.mean(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_segment, 
                sr=self.sr,
                hop_length=self.hop_length
            )
            features['spectral_rolloff'] = np.mean(spectral_rolloff)
            
            # 4. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                audio_segment,
                hop_length=self.hop_length
            )
            features['zcr'] = np.mean(zcr)
            
            # 5. Tempo and beat features (with error handling)
            try:
                tempo, beats = librosa.beat.beat_track(
                    y=audio_segment, 
                    sr=self.sr,
                    hop_length=self.hop_length
                )
                features['tempo'] = float(tempo) if tempo is not None else 120.0
            except Exception as e:
                print(f"Tempo extraction failed: {e}")
                features['tempo'] = 120.0  # Default tempo
            
            # 6. RMS Energy
            rms = librosa.feature.rms(
                y=audio_segment,
                hop_length=self.hop_length
            )
            features['rms'] = np.mean(rms)
            
            # 7. Spectral bandwidth (additional feature for better discrimination)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_segment,
                sr=self.sr,
                hop_length=self.hop_length
            )
            features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
            
        return features
    
    def features_to_vector(self, features: Dict) -> np.ndarray:
        """Convert feature dict to numerical vector"""
        if features is None:
            return np.zeros(40, dtype=np.float32)  # Updated size for new features
            
        vector = []
        
        # Helper function to safely add features to vector
        def add_feature(feature_name, default_shape):
            feature_data = features.get(feature_name, np.zeros(default_shape))
            if np.isscalar(feature_data):
                vector.append(float(feature_data))
            elif hasattr(feature_data, '__len__'):
                if hasattr(feature_data, 'flatten'):
                    vector.extend(feature_data.flatten().astype(float))
                else:
                    vector.extend([float(x) for x in feature_data])
            else:
                vector.append(float(feature_data))
        
        # Add MFCC features
        add_feature('mfcc_mean', 13)
        add_feature('mfcc_std', 13)
        add_feature('chroma_mean', 12)
        
        # Add scalar features
        scalar_features = [
            'spectral_centroid', 'spectral_rolloff', 'zcr', 
            'tempo', 'rms', 'spectral_bandwidth'
        ]
        
        for feature_name in scalar_features:
            feature_value = features.get(feature_name, 0.0)
            vector.append(float(feature_value))
        
        # Ensure consistent vector length
        vector_array = np.array(vector, dtype=np.float32)
        
        # Pad or truncate to fixed size
        target_size = 40
        if len(vector_array) < target_size:
            vector_array = np.pad(vector_array, (0, target_size - len(vector_array)))
        elif len(vector_array) > target_size:
            vector_array = vector_array[:target_size]
        
        # Handle NaN and infinite values
        vector_array = np.nan_to_num(vector_array, nan=0.0, posinf=1000.0, neginf=-1000.0)
        
        return vector_array


class MusicDatabase:
    """Store and search existing music fingerprints"""
    
    def __init__(self, db_path="music_db.pkl"):
        self.db_path = db_path
        self.fingerprints = {}  # {song_id: {'vector': np.array, 'metadata': dict}}
        self.fingerprinter = AudioFingerprinter()
        self.load_database()
    
    def add_song(self, song_id: str, audio_file: str, metadata: Dict = None):
        """Add a song to the database"""
        if not os.path.exists(audio_file):
            print(f"Error: Audio file {audio_file} not found")
            return False
            
        try:
            print(f"Loading audio file: {audio_file}")
            audio, sr = librosa.load(audio_file, sr=22050)
            
            if len(audio) == 0:
                print(f"Error: Empty audio file {audio_file}")
                return False
            
            # Process song in chunks to get multiple fingerprints
            chunk_size = sr * 5  # 5-second chunks
            vectors = []
            
            # Ensure we have enough audio for at least one chunk
            if len(audio) < chunk_size:
                # Use the entire audio if it's shorter than chunk_size
                features = self.fingerprinter.extract_features(audio)
                if features:
                    vector = self.fingerprinter.features_to_vector(features)
                    vectors.append(vector)
            else:
                # Process in overlapping chunks
                hop_size = chunk_size // 2
                for i in range(0, len(audio) - chunk_size + 1, hop_size):
                    chunk = audio[i:i + chunk_size]
                    features = self.fingerprinter.extract_features(chunk)
                    if features:
                        vector = self.fingerprinter.features_to_vector(features)
                        vectors.append(vector)
            
            if vectors:
                # Use mean vector as song representation
                song_vector = np.mean(vectors, axis=0)
                self.fingerprints[song_id] = {
                    'vector': song_vector,
                    'metadata': metadata or {},
                    'chunks': vectors[:10]  # Store max 10 chunks to save memory
                }
                self.save_database()
                print(f"Successfully added song: {song_id} ({len(vectors)} chunks processed)")
                return True
            else:
                print(f"Error: No valid features extracted from {audio_file}")
                return False
                
        except Exception as e:
            print(f"Error adding song {song_id}: {e}")
            return False
    
    def find_similar(self, query_vector: np.ndarray, threshold: float = 0.7) -> List[Tuple[str, float, Dict]]:
        """Find similar songs based on cosine similarity"""
        if query_vector is None or len(query_vector) == 0:
            return []
            
        similarities = []
        
        for song_id, data in self.fingerprints.items():
            if not isinstance(data, dict) or 'vector' not in data:
                continue
                
            try:
                stored_vector = data['vector']
                
                # Ensure vectors are same length
                if len(query_vector) != len(stored_vector):
                    min_len = min(len(query_vector), len(stored_vector))
                    query_vec = query_vector[:min_len]
                    stored_vec = stored_vector[:min_len]
                else:
                    query_vec = query_vector
                    stored_vec = stored_vector
                
                # Check for zero vectors
                if np.allclose(query_vec, 0) or np.allclose(stored_vec, 0):
                    continue
                
                # Calculate cosine similarity (1 - cosine_distance)
                try:
                    similarity = 1 - cosine(query_vec, stored_vec)
                    
                    # Handle NaN results
                    if np.isnan(similarity):
                        continue
                        
                    if similarity > threshold:
                        similarities.append((song_id, similarity, data.get('metadata', {})))
                        
                except Exception as e:
                    print(f"Similarity calculation error for {song_id}: {e}")
                    continue
                    
            except Exception as e:
                print(f"Error processing song {song_id}: {e}")
                continue
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def save_database(self):
        """Save database to disk"""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.fingerprints, f)
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def load_database(self):
        """Load database from disk"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    
                # Handle nested structure from old versions
                if isinstance(data, dict) and "fingerprints" in data:
                    self.fingerprints = data["fingerprints"]
                else:
                    self.fingerprints = data
                    
                print(f"Loaded {len(self.fingerprints)} songs from database")
            except Exception as e:
                print(f"Error loading database: {e}")
                self.fingerprints = {}
        else:
            print("No existing database found, starting fresh")


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
        self.detection_thread = None
        
        # Similarity detection settings
        self.similarity_threshold = 0.65  # Lowered threshold
        self.check_interval = 1.0  # Check every 1 second
        self.last_detection_time = 0
        self.detection_cooldown = 5.0  # 5 seconds between detections
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for real-time audio capture"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Add new audio data to buffer
        try:
            if len(indata.shape) > 1:
                audio_data = indata[:, 0]  # Take first channel
            else:
                audio_data = indata
            
            # Convert to float32 and normalize
            audio_data = audio_data.astype(np.float32)
            self.audio_buffer.extend(audio_data)
            
        except Exception as e:
            print(f"Audio callback error: {e}")
    
    def start_monitoring(self, similarity_callback: Optional[Callable] = None):
        """Start real-time similarity monitoring"""
        self.is_recording = True
        
        def detection_loop():
            print("Detection loop started")
            while self.is_recording:
                try:
                    current_time = time.time()
                    
                    # Check if we have enough audio data
                    if len(self.audio_buffer) >= self.buffer_size:
                        # Convert buffer to numpy array
                        audio_segment = np.array(list(self.audio_buffer), dtype=np.float32)
                        
                        # Check if audio has sufficient energy (not silence)
                        if np.max(np.abs(audio_segment)) < 0.001:
                            time.sleep(self.check_interval)
                            continue
                        
                        # Extract features and check similarity
                        features = self.fingerprinter.extract_features(audio_segment)
                        if features:
                            query_vector = self.fingerprinter.features_to_vector(features)
                            similar_songs = self.database.find_similar(
                                query_vector, 
                                threshold=self.similarity_threshold
                            )
                            
                            # Only report if we found matches and enough time has passed
                            if (similar_songs and 
                                current_time - self.last_detection_time > self.detection_cooldown):
                                
                                self.last_detection_time = current_time
                                if similarity_callback:
                                    similarity_callback(similar_songs)
                                else:
                                    print(f"Found {len(similar_songs)} similar songs")
                                    for song_id, similarity, metadata in similar_songs[:3]:
                                        print(f"  {song_id}: {similarity:.3f} similarity")
                    
                    time.sleep(self.check_interval)
                    
                except Exception as e:
                    print(f"Detection loop error: {e}")
                    time.sleep(self.check_interval)
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        # Start audio stream
        try:
            print(f"Starting audio stream (sample rate: {self.sample_rate})")
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=1024,
                dtype=np.float32
            ):
                print("Real-time monitoring started. Press Ctrl+C to stop.")
                while self.is_recording:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nStopping monitoring...")
            self.stop_monitoring()
        except Exception as e:
            print(f"Error in audio stream: {e}")
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_recording = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)
        print("Monitoring stopped")


def main():
    """Example usage and testing"""
    print("Music Similarity Engine - Debug Version")
    print("=" * 40)
    
    # Create database
    db = MusicDatabase()
    
    # Example: Add songs to database (uncomment and modify paths as needed)
    # db.add_song("song1", "path/to/your/song1.mp3", {"artist": "Artist 1", "genre": "Rock"})
    # db.add_song("song2", "path/to/your/song2.mp3", {"artist": "Artist 2", "genre": "Pop"})
    
    print(f"Database contains {len(db.fingerprints)} songs")
    
    if len(db.fingerprints) == 0:
        print("No songs in database. Add songs using db.add_song() before monitoring.")
        return
    
    # Start real-time detection
    detector = RealTimeSimilarityDetector(db)
    
    def on_similarity_found(similar_songs):
        print(f"\n🎵 Similarity detected! Found {len(similar_songs)} matches:")
        for song_id, similarity, metadata in similar_songs[:3]:  # Show top 3
            print(f"  📀 {song_id}: {similarity:.3f} similarity")
            if metadata:
                artist = metadata.get('artist', 'Unknown')
                genre = metadata.get('genre', 'Unknown')
                print(f"     Artist: {artist}, Genre: {genre}")
        print("-" * 40)
    
    try:
        detector.start_monitoring(similarity_callback=on_similarity_found)
    except Exception as e:
        print(f"Error during monitoring: {e}")


if __name__ == "__main__":
    main()