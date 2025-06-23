# import librosa
# import numpy as np
# import sounddevice as sd
# from scipy.spatial.distance import cosine
# from collections import deque
# import threading
# import time
# from typing import List, Dict, Tuple
# import pickle
# import os

# class AudioFingerprinter:
#     """Extract audio features for similarity comparison"""
    
#     def __init__(self, sr=22050, n_fft=2048, hop_length=512):
#         self.sr = sr
#         self.n_fft = n_fft
#         self.hop_length = hop_length
    
#     def extract_features(self, audio_segment: np.ndarray) -> Dict:
#         """Extract comprehensive audio features"""
        
#         # Ensure audio is mono and float32
#         if len(audio_segment.shape) > 1:
#             audio_segment = librosa.to_mono(audio_segment.T)
        
#         features = {}
        
#         try:
#             # 1. MFCCs (Mel-frequency cepstral coefficients)
#             mfcc = librosa.feature.mfcc(y=audio_segment, sr=self.sr, n_mfcc=13)
#             features['mfcc_mean'] = np.mean(mfcc, axis=1)
#             features['mfcc_std'] = np.std(mfcc, axis=1)
            
#             # 2. Chroma features (pitch class profiles)
#             chroma = librosa.feature.chroma_stft(y=audio_segment, sr=self.sr)
#             features['chroma_mean'] = np.mean(chroma, axis=1)
            
#             # 3. Spectral features
#             spectral_centroids = librosa.feature.spectral_centroid(y=audio_segment, sr=self.sr)
#             features['spectral_centroid'] = np.mean(spectral_centroids)
            
#             spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=self.sr)
#             features['spectral_rolloff'] = np.mean(spectral_rolloff)
            
#             # 4. Zero crossing rate
#             zcr = librosa.feature.zero_crossing_rate(audio_segment)
#             features['zcr'] = np.mean(zcr)
            
#             # 5. Tempo and beat features
#             tempo, beats = librosa.beat.beat_track(y=audio_segment, sr=self.sr)
#             features['tempo'] = tempo
            
#             # 6. RMS Energy
#             rms = librosa.feature.rms(y=audio_segment)
#             features['rms'] = np.mean(rms)
            
#         except Exception as e:
#             print(f"Feature extraction error: {e}")
#             return None
            
#         return features
    
#     def features_to_vector(self, features: Dict) -> np.ndarray:
#         """Convert feature dict to numerical vector"""
#         if features is None:
#             return np.zeros(32)  # Return zero vector on error
            
#         vector = []
#         vector.extend(features.get('mfcc_mean', np.zeros(13)))
#         vector.extend(features.get('mfcc_std', np.zeros(13)))
#         vector.extend(features.get('chroma_mean', np.zeros(12)))
#         vector.append(features.get('spectral_centroid', 0))
#         vector.append(features.get('spectral_rolloff', 0))
#         vector.append(features.get('zcr', 0))
#         vector.append(features.get('tempo', 0))
#         vector.append(features.get('rms', 0))
        
#         return np.array(vector, dtype=np.float32)


# class MusicDatabase:
#     """Store and search existing music fingerprints"""
    
#     def __init__(self, db_path="music_db.pkl"):
#         self.db_path = db_path
#         self.fingerprints = {}  # {song_id: {'vector': np.array, 'metadata': dict}}
#         self.load_database()
    
#     def add_song(self, song_id: str, audio_file: str, metadata: Dict = None):
#         """Add a song to the database"""
#         try:
#             audio, sr = librosa.load(audio_file, sr=22050)
#             fingerprinter = AudioFingerprinter()
            
#             # Process song in chunks to get multiple fingerprints
#             chunk_size = sr * 5  # 5-second chunks
#             vectors = []
            
#             for i in range(0, len(audio) - chunk_size, chunk_size // 2):
#                 chunk = audio[i:i + chunk_size]
#                 features = fingerprinter.extract_features(chunk)
#                 if features:
#                     vector = fingerprinter.features_to_vector(features)
#                     vectors.append(vector)
            
#             if vectors:
#                 # Use mean vector as song representation
#                 song_vector = np.mean(vectors, axis=0)
#                 self.fingerprints[song_id] = {
#                     'vector': song_vector,
#                     'metadata': metadata or {},
#                     'chunks': vectors  # Store individual chunks for detailed comparison
#                 }
#                 self.save_database()
#                 print(f"Added song: {song_id}")
                
#         except Exception as e:
#             print(f"Error adding song {song_id}: {e}")
    
#     def find_similar(self, query_vector: np.ndarray, threshold: float = 0.8) -> List[Tuple[str, float]]:
#         """Find similar songs based on cosine similarity"""
#         similarities = []
        
#         for song_id, data in self.fingerprints.items():
#             stored_vector = data['vector']
            
#             # Calculate cosine similarity (1 - cosine_distance)
#             similarity = 1 - cosine(query_vector, stored_vector)
            
#             if similarity > threshold:
#                 similarities.append((song_id, similarity, data['metadata']))
        
#         # Sort by similarity (highest first)
#         similarities.sort(key=lambda x: x[1], reverse=True)
#         return similarities
    
#     def save_database(self):
#         """Save database to disk"""
#         with open(self.db_path, 'wb') as f:
#             pickle.dump(self.fingerprints, f)
    
#     def load_database(self):
#         """Load database from disk"""
#         if os.path.exists(self.db_path):
#             try:
#                 with open(self.db_path, 'rb') as f:
#                     self.fingerprints = pickle.load(f)
#                 print(f"Loaded {len(self.fingerprints)} songs from database")
#             except Exception as e:
#                 print(f"Error loading database: {e}")


# class RealTimeSimilarityDetector:
#     """Real-time audio capture and similarity detection"""
    
#     def __init__(self, database: MusicDatabase, sample_rate=22050):
#         self.database = database
#         self.sample_rate = sample_rate
#         self.fingerprinter = AudioFingerprinter(sr=sample_rate)
        
#         # Audio buffer for real-time processing
#         self.buffer_size = sample_rate * 5  # 5 seconds
#         self.audio_buffer = deque(maxlen=self.buffer_size)
#         self.is_recording = False
        
#         # Similarity detection settings
#         self.similarity_threshold = 0.75
#         self.check_interval = 2.0  # Check every 2 seconds
        
#     def audio_callback(self, indata, frames, time, status):
#         """Callback for real-time audio capture"""
#         if status:
#             print(f"Audio callback status: {status}")
        
#         # Add new audio data to buffer
#         audio_data = indata[:, 0] if len(indata.shape) > 1 else indata
#         self.audio_buffer.extend(audio_data)
    
#     def start_monitoring(self, similarity_callback=None):
#         """Start real-time similarity monitoring"""
#         self.is_recording = True
        
#         def detection_loop():
#             while self.is_recording:
#                 if len(self.audio_buffer) >= self.buffer_size:
#                     # Convert buffer to numpy array
#                     audio_segment = np.array(list(self.audio_buffer))
                    
#                     # Extract features and check similarity
#                     features = self.fingerprinter.extract_features(audio_segment)
#                     if features:
#                         query_vector = self.fingerprinter.features_to_vector(features)
#                         similar_songs = self.database.find_similar(
#                             query_vector, 
#                             threshold=self.similarity_threshold
#                         )
                        
#                         if similar_songs and similarity_callback:
#                             similarity_callback(similar_songs)
                
#                 time.sleep(self.check_interval)
        
#         # Start audio stream
#         with sd.InputStream(
#             callback=self.audio_callback,
#             channels=1,
#             samplerate=self.sample_rate,
#             blocksize=1024
#         ):
#             print("Starting real-time similarity detection...")
#             print("Monitoring audio input...")
            
#             # Start detection in separate thread
#             detection_thread = threading.Thread(target=detection_loop)
#             detection_thread.daemon = True
#             detection_thread.start()
            
#             try:
#                 while self.is_recording:
#                     time.sleep(0.1)
#             except KeyboardInterrupt:
#                 print("\nStopping detection...")
#                 self.is_recording = False


# # Example usage and testing
# def similarity_found_callback(similar_songs):
#     """Callback function when similarities are detected"""
#     print("\n🎵 SIMILARITY DETECTED!")
#     for song_id, similarity, metadata in similar_songs[:3]:  # Show top 3
#         print(f"  📀 {song_id}: {similarity:.2%} similar")
#         if metadata:
#             print(f"     Artist: {metadata.get('artist', 'Unknown')}")
#             print(f"     Genre: {metadata.get('genre', 'Unknown')}")


# if __name__ == "__main__":
#     # Initialize system
#     print("Initializing Music Similarity Detection System...")
    
#     # Create database
#     db = MusicDatabase()
    
#     # Example: Add some songs to database (you'll need actual audio files)
#     # db.add_song("song1", "path/to/song1.mp3", {"artist": "Artist1", "genre": "Pop"})
#     # db.add_song("song2", "path/to/song2.mp3", {"artist": "Artist2", "genre": "Rock"})
    
#     # Start real-time detection
#     detector = RealTimeSimilarityDetector(db)
    
#     print("Ready to detect similarities!")
#     print("Play some music near your microphone...")
#     print("Press Ctrl+C to stop")
    
#     try:
#         detector.start_monitoring(similarity_callback=similarity_found_callback)
#     except KeyboardInterrupt:
#         print("\nShutting down...")

import librosa
import numpy as np
import sounddevice as sd
from scipy.spatial.distance import cosine
from collections import deque
import threading
import time
from typing import List, Dict, Tuple, Optional
import pickle
import os
import hashlib
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioFingerprinter:
    """Enhanced audio fingerprinting with improved feature extraction"""
    
    def __init__(self, sr=22050, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_audio_length = sr * 2  # Minimum 2 seconds
    
    def preprocess_audio(self, audio_segment: np.ndarray) -> np.ndarray:
        """Preprocess audio with normalization and filtering"""
        
        # Ensure audio is mono
        if len(audio_segment.shape) > 1:
            audio_segment = librosa.to_mono(audio_segment.T)
        
        # Normalize audio
        if np.max(np.abs(audio_segment)) > 0:
            audio_segment = audio_segment / np.max(np.abs(audio_segment))
        
        # Apply pre-emphasis filter
        pre_emphasis = 0.97
        audio_segment = np.append(audio_segment[0], audio_segment[1:] - pre_emphasis * audio_segment[:-1])
        
        return audio_segment
    
    def extract_features(self, audio_segment: np.ndarray) -> Optional[Dict]:
        """Extract comprehensive audio features with error handling"""
        
        # Check minimum length
        if len(audio_segment) < self.min_audio_length:
            logger.warning(f"Audio segment too short: {len(audio_segment)} samples")
            return None
        
        # Preprocess audio
        audio_segment = self.preprocess_audio(audio_segment)
        
        features = {}
        
        try:
            # 1. MFCCs with deltas
            mfcc = librosa.feature.mfcc(y=audio_segment, sr=self.sr, n_mfcc=13)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            features['mfcc_delta_mean'] = np.mean(mfcc_delta, axis=1)
            features['mfcc_delta2_mean'] = np.mean(mfcc_delta2, axis=1)
            
            # 2. Enhanced Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_segment, sr=self.sr)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            features['chroma_std'] = np.std(chroma, axis=1)
            
            # 3. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_segment, sr=self.sr)
            features['spectral_centroid'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=self.sr)
            features['spectral_rolloff'] = np.mean(spectral_rolloff)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_segment, sr=self.sr)
            features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
            
            # 4. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_segment)
            features['zcr'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # 5. Tempo and beat features
            try:
                tempo, beats = librosa.beat.beat_track(y=audio_segment, sr=self.sr)
                features['tempo'] = float(tempo)
                features['beat_strength'] = np.mean(librosa.onset.onset_strength(y=audio_segment, sr=self.sr))
            except:
                features['tempo'] = 0.0
                features['beat_strength'] = 0.0
            
            # 6. RMS Energy
            rms = librosa.feature.rms(y=audio_segment)
            features['rms'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            # 7. Tonal features
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_segment), sr=self.sr)
            features['tonnetz_mean'] = np.mean(tonnetz, axis=1)
            
            # 8. Mel-spectrogram features
            mel_spec = librosa.feature.melspectrogram(y=audio_segment, sr=self.sr)
            features['mel_spec_mean'] = np.mean(mel_spec, axis=1)[:20]  # First 20 mel bands
            
            # 9. Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio_segment, sr=self.sr)
            features['spectral_contrast'] = np.mean(contrast, axis=1)
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None
            
        return features
    
    def features_to_vector(self, features: Dict) -> np.ndarray:
        """Convert feature dict to numerical vector with improved organization"""
        if features is None:
            return np.zeros(128)  # Increased vector size for more features
            
        vector = []
        
        # MFCC features (52 values: 13*4)
        vector.extend(features.get('mfcc_mean', np.zeros(13)))
        vector.extend(features.get('mfcc_std', np.zeros(13)))
        vector.extend(features.get('mfcc_delta_mean', np.zeros(13)))
        vector.extend(features.get('mfcc_delta2_mean', np.zeros(13)))
        
        # Chroma features (24 values: 12*2)
        vector.extend(features.get('chroma_mean', np.zeros(12)))
        vector.extend(features.get('chroma_std', np.zeros(12)))
        
        # Spectral features (4 values)
        vector.append(features.get('spectral_centroid', 0))
        vector.append(features.get('spectral_centroid_std', 0))
        vector.append(features.get('spectral_rolloff', 0))
        vector.append(features.get('spectral_bandwidth', 0))
        
        # Temporal features (3 values)
        vector.append(features.get('zcr', 0))
        vector.append(features.get('zcr_std', 0))
        vector.append(features.get('tempo', 0))
        vector.append(features.get('beat_strength', 0))
        
        # Energy features (2 values)
        vector.append(features.get('rms', 0))
        vector.append(features.get('rms_std', 0))
        
        # Tonal features (6 values)
        vector.extend(features.get('tonnetz_mean', np.zeros(6)))
        
        # Mel-spectrogram features (20 values)
        vector.extend(features.get('mel_spec_mean', np.zeros(20)))
        
        # Spectral contrast (7 values)
        vector.extend(features.get('spectral_contrast', np.zeros(7)))
        
        return np.array(vector, dtype=np.float32)

    def get_audio_hash(self, audio_path: str) -> str:
        """Generate hash for audio file to avoid duplicates"""
        with open(audio_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()


class MusicDatabase:
    """Enhanced music database with improved search and management"""
    
    def __init__(self, db_path="music_db.pkl", backup_interval=10):
        self.db_path = Path(db_path)
        self.backup_interval = backup_interval
        self.fingerprints = {}
        self.audio_hashes = {}  # Track duplicate files
        self.song_counter = 0
        self.load_database()
    
    def add_song(self, song_id: str, audio_file: str, metadata: Dict = None, overwrite: bool = False):
        """Add a song to the database with duplicate detection"""
        try:
            fingerprinter = AudioFingerprinter()
            
            # Check for duplicates
            audio_hash = fingerprinter.get_audio_hash(audio_file)
            if audio_hash in self.audio_hashes and not overwrite:
                existing_song = self.audio_hashes[audio_hash]
                logger.warning(f"Duplicate audio detected. Existing song: {existing_song}")
                return False
            
            # Load and validate audio
            audio, sr = librosa.load(audio_file, sr=22050)
            if len(audio) < 22050 * 5:  # Minimum 5 seconds
                logger.warning(f"Audio file too short: {song_id}")
                return False
            
            # Process song in overlapping chunks
            chunk_size = sr * 10  # 10-second chunks
            overlap = sr * 5      # 5-second overlap
            vectors = []
            chunk_features = []
            
            for i in range(0, len(audio) - chunk_size, overlap):
                chunk = audio[i:i + chunk_size]
                features = fingerprinter.extract_features(chunk)
                if features:
                    vector = fingerprinter.features_to_vector(features)
                    vectors.append(vector)
                    chunk_features.append(features)
            
            if not vectors:
                logger.error(f"No valid features extracted for {song_id}")
                return False
            
            # Calculate representative vector
            song_vector = np.mean(vectors, axis=0)
            
            # Store comprehensive song data
            self.fingerprints[song_id] = {
                'vector': song_vector,
                'chunk_vectors': vectors,
                'chunk_features': chunk_features,
                'metadata': metadata or {},
                'audio_hash': audio_hash,
                'duration': len(audio) / sr,
                'chunk_count': len(vectors),
                'added_timestamp': time.time()
            }
            
            self.audio_hashes[audio_hash] = song_id
            self.song_counter += 1
            
            # Auto-backup
            if self.song_counter % self.backup_interval == 0:
                self.backup_database()
            
            self.save_database()
            logger.info(f"Added song: {song_id} ({len(vectors)} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Error adding song {song_id}: {e}")
            return False
    
    def find_similar(self, query_vector: np.ndarray, threshold: float = 0.8, max_results: int = 10) -> List[Tuple[str, float, Dict]]:
        """Enhanced similarity search with chunk-level analysis"""
        similarities = []
        
        for song_id, data in self.fingerprints.items():
            # Primary similarity using mean vector
            main_similarity = 1 - cosine(query_vector, data['vector'])
            
            # Chunk-level similarity for more accurate matching
            chunk_similarities = []
            for chunk_vector in data.get('chunk_vectors', []):
                chunk_sim = 1 - cosine(query_vector, chunk_vector)
                chunk_similarities.append(chunk_sim)
            
            # Calculate final similarity score
            if chunk_similarities:
                max_chunk_sim = max(chunk_similarities)
                avg_chunk_sim = np.mean(chunk_similarities)
                
                # Weighted combination
                final_similarity = (main_similarity * 0.4 + 
                                  max_chunk_sim * 0.4 + 
                                  avg_chunk_sim * 0.2)
            else:
                final_similarity = main_similarity
            
            if final_similarity > threshold:
                similarities.append((song_id, final_similarity, data['metadata']))
        
        # Sort by similarity and limit results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]
    
    def get_song_info(self, song_id: str) -> Optional[Dict]:
        """Get detailed information about a song"""
        return self.fingerprints.get(song_id)
    
    def remove_song(self, song_id: str) -> bool:
        """Remove a song from the database"""
        if song_id in self.fingerprints:
            song_data = self.fingerprints[song_id]
            audio_hash = song_data.get('audio_hash')
            
            del self.fingerprints[song_id]
            if audio_hash and audio_hash in self.audio_hashes:
                del self.audio_hashes[audio_hash]
            
            self.save_database()
            logger.info(f"Removed song: {song_id}")
            return True
        return False
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        if not self.fingerprints:
            return {'total_songs': 0, 'total_chunks': 0, 'avg_duration': 0}
        
        total_songs = len(self.fingerprints)
        total_chunks = sum(len(data.get('chunk_vectors', [])) for data in self.fingerprints.values())
        avg_duration = np.mean([data.get('duration', 0) for data in self.fingerprints.values()])
        
        return {
            'total_songs': total_songs,
            'total_chunks': total_chunks,
            'avg_duration': avg_duration,
            'db_size_mb': self.db_path.stat().st_size / (1024*1024) if self.db_path.exists() else 0
        }
    
    def backup_database(self):
        """Create backup of database"""
        if self.db_path.exists():
            backup_path = self.db_path.with_suffix(f'.backup_{int(time.time())}.pkl')
            self.db_path.rename(backup_path)
            logger.info(f"Database backed up to {backup_path}")
    
    def save_database(self):
        """Save database to disk with error handling"""
        try:
            temp_path = self.db_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                pickle.dump({
                    'fingerprints': self.fingerprints,
                    'audio_hashes': self.audio_hashes,
                    'version': '2.0'
                }, f)
            temp_path.replace(self.db_path)
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def load_database(self):
        """Load database from disk with migration support"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, dict) and 'version' in data:
                    # New format
                    self.fingerprints = data.get('fingerprints', {})
                    self.audio_hashes = data.get('audio_hashes', {})
                else:
                    # Legacy format
                    self.fingerprints = data
                    self.audio_hashes = {}
                
                logger.info(f"Loaded {len(self.fingerprints)} songs from database")
                
            except Exception as e:
                logger.error(f"Error loading database: {e}")
                self.fingerprints = {}
                self.audio_hashes = {}


class RealTimeSimilarityDetector:
    """Enhanced real-time detector with improved performance"""
    
    def __init__(self, database: MusicDatabase, sample_rate=22050):
        self.database = database
        self.sample_rate = sample_rate
        self.fingerprinter = AudioFingerprinter(sr=sample_rate)
        
        # Audio processing settings
        self.buffer_size = sample_rate * 8  # 8 seconds
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.processing_queue = deque(maxlen=5)
        
        # Detection settings
        self.similarity_threshold = 0.75
        self.check_interval = 1.5
        self.min_match_duration = 3  # Minimum seconds for valid match
        
        # State tracking
        self.is_recording = False
        self.last_matches = {}
        self.match_history = deque(maxlen=10)
        
    def audio_callback(self, indata, frames, time, status):
        """Optimized audio callback"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Add audio data to buffer
        audio_data = indata[:, 0] if len(indata.shape) > 1 else indata
        self.audio_buffer.extend(audio_data.flatten())
    
    def process_audio_chunk(self):
        """Process audio chunk for similarity detection"""
        if len(self.audio_buffer) < self.buffer_size:
            return None
        
        # Get audio segment
        audio_segment = np.array(list(self.audio_buffer))
        
        # Extract features
        features = self.fingerprinter.extract_features(audio_segment)
        if not features:
            return None
        
        query_vector = self.fingerprinter.features_to_vector(features)
        
        # Find similarities
        similar_songs = self.database.find_similar(
            query_vector, 
            threshold=self.similarity_threshold,
            max_results=5
        )
        
        return similar_songs
    
    def start_monitoring(self, similarity_callback=None):
        """Start enhanced real-time monitoring"""
        self.is_recording = True
        
        def detection_loop():
            while self.is_recording:
                try:
                    similar_songs = self.process_audio_chunk()
                    
                    if similar_songs and similarity_callback:
                        # Filter and validate matches
                        valid_matches = self.validate_matches(similar_songs)
                        if valid_matches:
                            similarity_callback(valid_matches)
                    
                    time.sleep(self.check_interval)
                    
                except Exception as e:
                    logger.error(f"Detection loop error: {e}")
                    time.sleep(1)
        
        # Start audio stream with optimized settings
        try:
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=2048,
                latency='low'
            ):
                logger.info("Starting enhanced real-time similarity detection...")
                
                # Start detection thread
                detection_thread = threading.Thread(target=detection_loop, daemon=True)
                detection_thread.start()
                
                while self.is_recording:
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Audio stream error: {e}")
            self.is_recording = False
    
    def validate_matches(self, matches: List[Tuple]) -> List[Tuple]:
        """Validate and filter matches to reduce false positives"""
        current_time = time.time()
        valid_matches = []
        
        for song_id, similarity, metadata in matches:
            # Check if this is a sustained match
            if song_id in self.last_matches:
                last_time, last_similarity = self.last_matches[song_id]
                
                # If match has been consistent, it's more likely valid
                if (current_time - last_time < self.min_match_duration and 
                    abs(similarity - last_similarity) < 0.1):
                    valid_matches.append((song_id, similarity, metadata))
            
            # Update last match time
            self.last_matches[song_id] = (current_time, similarity)
        
        # Clean old matches
        expired_keys = [k for k, (t, s) in self.last_matches.items() 
                       if current_time - t > self.min_match_duration * 2]
        for key in expired_keys:
            del self.last_matches[key]
        
        return valid_matches
    
    def stop_monitoring(self):
        """Stop monitoring and cleanup"""
        self.is_recording = False
        self.audio_buffer.clear()
        self.last_matches.clear()
        logger.info("Stopped real-time detection")


# Enhanced demo and testing
def enhanced_similarity_callback(similar_songs):
    """Enhanced callback with better formatting"""
    print("\n🎵 ENHANCED SIMILARITY DETECTED!")
    print("=" * 50)
    
    for i, (song_id, similarity, metadata) in enumerate(similar_songs[:3], 1):
        print(f"{i}. 📀 {song_id}")
        print(f"   🎯 Similarity: {similarity:.1%}")
        
        if metadata:
            if 'artist' in metadata:
                print(f"   🎤 Artist: {metadata['artist']}")
            if 'genre' in metadata:
                print(f"   🎼 Genre: {metadata['genre']}")
            if 'duration' in metadata:
                print(f"   ⏱️  Duration: {metadata['duration']:.1f}s")
        print()


if __name__ == "__main__":
    print("🎵 Enhanced Music Similarity Detection System")
    print("=" * 50)
    
    # Initialize enhanced system
    db = MusicDatabase()
    
    # Display database stats
    stats = db.get_database_stats()
    print(f"Database: {stats['total_songs']} songs, {stats['total_chunks']} chunks")
    print(f"Average duration: {stats['avg_duration']:.1f}s")
    print(f"Database size: {stats['db_size_mb']:.1f} MB")
    
    # Start enhanced detection
    detector = RealTimeSimilarityDetector(db)
    
    print("\n🎤 Ready for enhanced real-time detection!")
    print("Play music near your microphone...")
    print("Press Ctrl+C to stop")
    
    try:
        detector.start_monitoring(similarity_callback=enhanced_similarity_callback)
    except KeyboardInterrupt:
        print("\n👋 Shutting down enhanced system...")
        detector.stop_monitoring()