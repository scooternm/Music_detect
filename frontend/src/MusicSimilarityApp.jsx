import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Upload, Music, AlertCircle, CheckCircle, Trash2, Database } from 'lucide-react';

const MusicSimilarityApp = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [similarities, setSimilarities] = useState([]);
  const [databaseSongs, setDatabaseSongs] = useState([]);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [audioLevel, setAudioLevel] = useState(0);
  
  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);

  // Initialize WebSocket connection
  useEffect(() => {
    connectWebSocket();
    fetchDatabaseSongs();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  const connectWebSocket = () => {
    try {
      wsRef.current = new WebSocket('ws://localhost:8000/ws/realtime');
      
      wsRef.current.onopen = () => {
        setConnectionStatus('connected');
        console.log('WebSocket connected');
      };
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'similarity_detected') {
          setSimilarities(prev => {
            const newSimilarities = [...data.matches, ...prev];
            return newSimilarities.slice(0, 10); // Keep last 10 matches
          });
        }
      };
      
      wsRef.current.onclose = () => {
        setConnectionStatus('disconnected');
        setTimeout(connectWebSocket, 3000); // Reconnect after 3 seconds
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
      };
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      setConnectionStatus('error');
    }
  };

  const fetchDatabaseSongs = async () => {
    try {
      const response = await fetch('http://localhost:8000/database/songs');
      const data = await response.json();
      setDatabaseSongs(data.songs);
    } catch (error) {
      console.error('Failed to fetch database songs:', error);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Setup audio context for visualization
      audioContextRef.current = new AudioContext();
      analyserRef.current = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      
      analyserRef.current.fftSize = 256;
      const bufferLength = analyserRef.current.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      
      const updateAudioLevel = () => {
        analyserRef.current.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((sum, value) => sum + value, 0) / bufferLength;
        setAudioLevel(average / 255);
        
        if (isRecording) {
          animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
        }
      };
      updateAudioLevel();
      
      // Setup MediaRecorder for WebSocket streaming
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
          // Convert blob to base64 and send
          const reader = new FileReader();
          reader.onload = () => {
            const audioData = reader.result.split(',')[1]; // Remove data:audio/webm;base64,
            wsRef.current.send(JSON.stringify({
              type: 'audio_chunk',
              audio_data: audioData
            }));
          };
          reader.readAsDataURL(event.data);
        }
      };
      
      mediaRecorderRef.current.start(1000); // Send data every second
      setIsRecording(true);
      
    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Error accessing microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    
    setIsRecording(false);
    setAudioLevel(0);
  };

  const uploadSong = async (file, metadata = {}) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('metadata', JSON.stringify(metadata));
    
    try {
      setUploadProgress({ status: 'uploading', progress: 0 });
      
      const response = await fetch('http://localhost:8000/upload-song', {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        setUploadProgress({ status: 'success', progress: 100 });
        fetchDatabaseSongs();
        setTimeout(() => setUploadProgress(null), 3000);
      } else {
        throw new Error('Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      setUploadProgress({ status: 'error', progress: 0 });
      setTimeout(() => setUploadProgress(null), 3000);
    }
  };

  const deleteSong = async (songId) => {
    try {
      const response = await fetch(`http://localhost:8000/database/songs/${songId}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        fetchDatabaseSongs();
      }
    } catch (error) {
      console.error('Delete error:', error);
    }
  };

  const getSimilarityColor = (similarity) => {
    if (similarity > 0.9) return 'text-red-600 bg-red-50';
    if (similarity > 0.8) return 'text-orange-600 bg-orange-50';
    if (similarity > 0.7) return 'text-yellow-600 bg-yellow-50';
    return 'text-green-600 bg-green-50';
  };

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'text-green-600';
      case 'error': return 'text-red-600';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-pink-400 via-purple-400 to-indigo-400 bg-clip-text text-transparent">
            Music Similarity Detector
          </h1>
          <p className="text-xl text-gray-300 mb-4">
            Real-time audio similarity detection - Like Grammarly for music
          </p>
          <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full border ${getConnectionStatusColor()}`}>
            <div className={`w-2 h-2 rounded-full ${connectionStatus === 'connected' ? 'bg-green-500' : 'bg-gray-500'}`}></div>
            <span className="text-sm font-medium">
              {connectionStatus === 'connected' ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* Recording Section */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
              <Music className="text-purple-400" />
              Real-Time Detection
            </h2>
            
            {/* Recording Controls */}
            <div className="text-center mb-8">
              <button
                onClick={isRecording ? stopRecording : startRecording}
                className={`w-24 h-24 rounded-full flex items-center justify-center transition-all duration-300 ${
                  isRecording 
                    ? 'bg-red-500 hover:bg-red-600 animate-pulse' 
                    : 'bg-purple-500 hover:bg-purple-600'
                }`}
              >
                {isRecording ? <MicOff size={32} /> : <Mic size={32} />}
              </button>
              
              {/* Audio Level Visualization */}
              {isRecording && (
                <div className="mt-4">
                  <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
                    <div 
                      className="bg-gradient-to-r from-green-400 to-purple-500 h-2 rounded-full transition-all duration-100"
                      style={{ width: `${audioLevel * 100}%` }}
                    ></div>
                  </div>
                  <p className="text-sm text-gray-300">Audio Level: {Math.round(audioLevel * 100)}%</p>
                </div>
              )}
            </div>
            
            {/* Similarity Results */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Recent Matches</h3>
              {similarities.length === 0 ? (
                <p className="text-gray-400 text-center py-8">
                  {isRecording ? 'Listening for similarities...' : 'Start recording to detect similarities'}
                </p>
              ) : (
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {similarities.map((match, index) => (
                    <div key={index} className={`p-4 rounded-lg border ${getSimilarityColor(match.similarity)}`}>
                      <div className="flex justify-between items-start mb-2">
                        <h4 className="font-semibold text-gray-800">{match.song_id}</h4>
                        <span className="text-sm font-bold text-gray-800">
                          {Math.round(match.similarity * 100)}% match
                        </span>
                      </div>
                      {match.metadata && (
                        <div className="text-sm opacity-80 text-gray-700">
                          {match.metadata.artist && <p>Artist: {match.metadata.artist}</p>}
                          {match.metadata.genre && <p>Genre: {match.metadata.genre}</p>}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Database Management */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
              <Database className="text-blue-400" />
              Music Database
            </h2>
            
            {/* Upload Section */}
            <div className="mb-8">
              <h3 className="text-lg font-semibold mb-4">Add New Song</h3>
              <div className="border-2 border-dashed border-gray-400 rounded-lg p-6 text-center">
                <Upload className="mx-auto mb-4 text-gray-400" size={48} />
                <input
                  type="file"
                  accept="audio/*"
                  onChange={(e) => {
                    const file = e.target.files[0];
                    if (file) {
                      const metadata = {
                        artist: prompt('Artist name (optional):') || '',
                        genre: prompt('Genre (optional):') || '',
                        title: file.name.split('.')[0]
                      };
                      uploadSong(file, metadata);
                    }
                  }}
                  className="hidden"
                  id="song-upload"
                />
                <label htmlFor="song-upload" className="cursor-pointer">
                  <p className="text-gray-300 mb-2">Click to upload audio file</p>
                  <p className="text-sm text-gray-500">MP3, WAV, M4A supported</p>
                </label>
              </div>
              
              {/* Upload Progress */}
              {uploadProgress && (
                <div className="mt-4 p-3 rounded-lg bg-gray-800">
                  <div className="flex items-center gap-2">
                    {uploadProgress.status === 'success' ? (
                      <CheckCircle className="text-green-400" size={20} />
                    ) : uploadProgress.status === 'error' ? (
                      <AlertCircle className="text-red-400" size={20} />
                    ) : (
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-400"></div>
                    )}
                    <span className="text-sm">
                      {uploadProgress.status === 'success' ? 'Upload successful!' :
                       uploadProgress.status === 'error' ? 'Upload failed!' : 'Uploading...'}
                    </span>
                  </div>
                </div>
              )}
            </div>
            
            {/* Database Songs List */}
            <div>
              <h3 className="text-lg font-semibold mb-4">
                Database Songs ({databaseSongs.length})
              </h3>
              
              {databaseSongs.length === 0 ? (
                <p className="text-gray-400 text-center py-8">
                  No songs in database. Upload some songs to get started!
                </p>
              ) : (
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {databaseSongs.map((song, index) => (
                    <div key={index} className="bg-gray-800/50 p-4 rounded-lg flex justify-between items-center">
                      <div>
                        <h4 className="font-semibold">{song.song_id}</h4>
                        {song.metadata && (
                          <div className="text-sm text-gray-400">
                            {song.metadata.artist && <span>by {song.metadata.artist}</span>}
                            {song.metadata.genre && <span> • {song.metadata.genre}</span>}
                          </div>
                        )}
                        <p className="text-xs text-gray-500">
                          {song.features_count} feature chunks
                        </p>
                      </div>
                      <button
                        onClick={() => {
                          if (confirm(`Delete "${song.song_id}"?`)) {
                            deleteSong(song.song_id);
                          }
                        }}
                        className="text-red-400 hover:text-red-300 transition-colors"
                      >
                        <Trash2 size={20} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Instructions */}
        <div className="mt-12 bg-white/5 backdrop-blur-lg rounded-2xl p-8 border border-white/10">
          <h2 className="text-2xl font-bold mb-4">How to Use</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
            <div className="space-y-2">
              <h3 className="font-semibold text-purple-400">1. Build Your Database</h3>
              <p className="text-gray-300">
                Upload reference songs to your database. These will be used to detect similarities against.
              </p>
            </div>
            <div className="space-y-2">
              <h3 className="font-semibold text-blue-400">2. Start Detection</h3>
              <p className="text-gray-300">
                Click the microphone to start real-time detection. Play music near your device.
              </p>
            </div>
            <div className="space-y-2">
              <h3 className="font-semibold text-green-400">3. View Results</h3>
              <p className="text-gray-300">
                Similar songs will appear with similarity percentages. Higher percentages indicate closer matches.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MusicSimilarityApp;