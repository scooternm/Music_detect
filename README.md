# Music_detect
# Real-Time Music Detection System

A complete music similarity detection system with real-time audio monitoring and web interface.

## Features

- 🎵 **Upload songs** to build a fingerprint database
- 🎯 **Real-time detection** of similar music through microphone
- 📊 **Audio analysis** of uploaded files
- 🌐 **Web interface** for easy interaction
- 🔄 **WebSocket support** for real-time updates
- 📱 **Responsive design** that works on desktop and mobile

## Quick Start

### 1. Backend Setup (Python)

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install additional audio libraries (system-level)
# For Ubuntu/Debian:
sudo apt-get install libsndfile1 libportaudio2 ffmpeg

# For macOS:
brew install portaudio ffmpeg

# For Windows:
# Download and install FFmpeg from https://ffmpeg.org/
```

### 2. Start the Backend Server

```bash
# Run the FastAPI server
python music_similarity_api.py

# The API will be available at:
# - Main API: http://localhost:8000
# - Documentation: http://localhost:8000/docs
# - WebSocket: ws://localhost:8000/ws/realtime
```

### 3. Frontend Setup (React)

The React component is ready to use in the Claude interface above, but if you want to run it locally:

```bash
# Install Node.js dependencies
npm install

# Or create a new React project and use the component
npx create-react-app music-detection-frontend
cd music-detection-frontend
npm install lucide-react
# Copy the React component into src/App.js
npm start
```

## How to Use

### Building Your Music Database

1. **Upload Songs**: Click "Choose Audio File" to upload MP3, WAV, M4A, or FLAC files
2. **Database Management**: View uploaded songs and delete them if needed
3. **Supported Formats**: MP3, WAV, M4A, FLAC

### Analyzing Audio Files

1. **Single File Analysis**: Upload a file using "Analyze Audio File" button
2. **Batch Analysis**: Use the API endpoint `/batch-analyze` for multiple files
3. **View Results**: See similarity scores and matching songs

### Real-Time Detection

1. **Start Listening**: Click "Start Listening" to begin microphone monitoring
2. **Play Music**: Play songs near your microphone
3. **Live Matches**: See real-time matches with similarity percentages
4. **Auto-Clear**: Matches automatically clear after 5 seconds

## API Endpoints

### Core Endpoints

- `POST /upload-song` - Upload a song to the database
- `POST /analyze-audio` - Analyze an audio file for similarities
- `GET /database/songs` - List all songs in database
- `DELETE /database/songs/{song_id}` - Delete a song
- `POST /batch-analyze` - Analyze multiple files
- `WS /ws/realtime` - WebSocket for real-time detection

### Health & Status

- `GET /health` - Health check and system status
- `GET /` - API information

## Configuration

### Audio Settings

The system uses these default audio parameters:
- **Sample Rate**: 22,050 Hz
- **Chunk Size**: 5 seconds for processing
- **Similarity Threshold**: 75% for real-time, 70% for file analysis
- **Features**: MFCC, Chroma, Spectral features, Tempo, RMS

### Customization

You can modify these settings in `music_similarity_engine.py`:

```python
# Adjust similarity thresholds
self.similarity_threshold = 0.75  # Real-time detection
threshold=0.7  # File analysis

# Modify audio processing parameters
AudioFingerprinter(sr=22050, n_fft=2048, hop_length=512)
```

## Troubleshooting

### Common Issues

1. **Microphone Access Denied**
   - Enable microphone permissions in your browser
   - Check browser security settings for localhost

2. **WebSocket Connection Failed**
   - Ensure the backend server is running on port 8000
   - Check firewall settings

3. **Audio Processing Errors**
   - Install required audio libraries (FFmpeg, PortAudio)
   - Check audio file formats are supported

4. **No Similarities Found**
   - Ensure you have songs in your database
   - Try lowering the similarity threshold
   - Check audio quality and length

### Performance Tips

1. **Database Size**: Large databases may slow down similarity searches
2. **Audio Quality**: Higher quality audio provides better fingerprinting
3. **Chunk Size**: Smaller chunks = more responsive but less accurate
4. **Real-time Processing**: Close unnecessary applications for better performance

## Technical Details

### Audio Fingerprinting

The system extracts these features from audio:
- **MFCC** (Mel-frequency cepstral coefficients) - 13 coefficients
- **Chroma** features - 12 pitch classes
- **Spectral** features - centroid, rolloff
- **Temporal** features - zero crossing rate, tempo, RMS energy

### Similarity Calculation

- Uses **cosine similarity** between feature vectors
- Processes audio in 5-second overlapping chunks
- Combines multiple chunk similarities for final score

### Database Storage

- Stores fingerprints in pickle format (`music_db.pkl`)
- Each song has multiple chunk representations
- Metadata stored alongside fingerprints

## Extensions & Improvements

### Possible Enhancements

1. **Advanced Features**:
   - Beat tracking and rhythm analysis
   - Harmonic analysis
   - Onset detection

2. **UI Improvements**:
   - Waveform visualization
   - Real-time audio level meters
   - Advanced filtering and search

3. **Performance**:
   - Database indexing for faster searches
   - GPU acceleration for feature extraction
   - Distributed processing for large databases

4. **Additional Formats**:
   - YouTube URL analysis
   - Streaming service integration
   - MIDI file support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is open source and available under the MIT License.