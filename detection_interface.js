import React, { useState, useRef, useEffect } from 'react';
import { Upload, Mic, MicOff, Music, Database, Play, Pause, Trash2, Volume2, VolumeX } from 'lucide-react';

const MusicDetectionApp = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [uploadedSongs, setUploadedSongs] = useState([]);
  const [similarityResults, setSimilarityResults] = useState([]);
  const [realtimeMatches, setRealtimeMatches] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [audioContext, setAudioContext] = useState(null);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [audioChunks, setAudioChunks] = useState([]);
  
  const wsRef = useRef(null);
  const fileInputRef = useRef(null);
  const analyzeInputRef = useRef(null);

  // Initialize WebSocket connection
  useEffect(() => {
    connectWebSocket();
    fetchDatabaseSongs();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
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
          setRealtimeMatches(data.matches);
          // Auto-clear matches after 5 seconds
          setTimeout(() => setRealtimeMatches([]), 5000);
        }
      };
      
      wsRef.current.onclose = () => {
        setConnectionStatus('disconnected');
        // Attempt to reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
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
      setUploadedSongs(data.songs || []);
    } catch (error) {
      console.error('Failed to fetch database songs:', error);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('song_id', file.name.split('.')[0]);
    formData.append('metadata', JSON.stringify({
      filename: file.name,
      size: file.size,
      uploaded_at: new Date().toISOString()
    }));

    try {
      const response = await fetch('http://localhost:8000/upload-song', {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        fetchDatabaseSongs();
        alert('Song uploaded successfully!');
      } else {
        alert('Failed to upload song');
      }
    } catch (error) {
      console.error('Upload error:', error);
      alert('Upload failed');
    }
  };

  const handleAnalyzeAudio = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/analyze-audio', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      setSimilarityResults(data.similar_songs || []);
    } catch (error) {
      console.error('Analysis error:', error);
      alert('Analysis failed');
    }
  };

  const startRealtimeDetection = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('Microphone access not supported');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 22050,
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false
        } 
      });

      const context = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 22050
      });
      
      const recorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      setAudioContext(context);
      setMediaRecorder(recorder);
      setIsRecording(true);

      // Process audio in chunks
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          // Convert to base64 and send via WebSocket
          const reader = new FileReader();
          reader.onload = () => {
            const base64Audio = reader.result.split(',')[1];
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify({
                type: 'audio_chunk',
                audio_data: base64Audio
              }));
            }
          };
          reader.readAsDataURL(event.data);
        }
      };

      recorder.start(1000); // Record in 1-second chunks
      
    } catch (error) {
      console.error('Failed to start recording:', error);
      alert('Failed to access microphone');
    }
  };

  const stopRealtimeDetection = () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
    
    if (audioContext) {
      audioContext.close();
    }
    
    setIsRecording(false);
    setMediaRecorder(null);
    setAudioContext(null);
    setRealtimeMatches([]);
  };

  const deleteSong = async (songId) => {
    try {
      const response = await fetch(`http://localhost:8000/database/songs/${songId}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        fetchDatabaseSongs();
      } else {
        alert('Failed to delete song');
      }
    } catch (error) {
      console.error('Delete error:', error);
    }
  };

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'text-green-500';
      case 'disconnected': return 'text-red-500';
      case 'error': return 'text-orange-500';
      default: return 'text-gray-500';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-pink-400 to-purple-400 bg-clip-text text-transparent">
            🎵 Real-Time Music Detection
          </h1>
          <p className="text-gray-300">Upload songs to build your database, then detect similarities in real-time</p>
          <div className={`inline-flex items-center mt-2 ${getConnectionStatusColor()}`}>
            <div className={`w-2 h-2 rounded-full mr-2 ${connectionStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'}`}></div>
            Connection: {connectionStatus}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Upload & Database */}
          <div className="space-y-6">
            {/* Upload Section */}
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
              <h2 className="text-2xl font-semibold mb-4 flex items-center">
                <Upload className="mr-2" size={24} />
                Build Music Database
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Upload Song to Database</label>
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileUpload}
                    accept=".mp3,.wav,.m4a,.flac"
                    className="hidden"
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white py-3 px-4 rounded-lg transition-all duration-200 flex items-center justify-center"
                  >
                    <Music className="mr-2" size={20} />
                    Choose Audio File
                  </button>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">Analyze Audio for Similarities</label>
                  <input
                    type="file"
                    ref={analyzeInputRef}
                    onChange={handleAnalyzeAudio}
                    accept=".mp3,.wav,.m4a,.flac"
                    className="hidden"
                  />
                  <button
                    onClick={() => analyzeInputRef.current?.click()}
                    className="w-full bg-gradient-to-r from-blue-500 to-teal-500 hover:from-blue-600 hover:to-teal-600 text-white py-3 px-4 rounded-lg transition-all duration-200 flex items-center justify-center"
                  >
                    <Database className="mr-2" size={20} />
                    Analyze Audio File
                  </button>
                </div>
              </div>
            </div>

            {/* Database Songs */}
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
              <h3 className="text-xl font-semibold mb-4 flex items-center">
                <Database className="mr-2" size={20} />
                Database Songs ({uploadedSongs.length})
              </h3>
              
              <div className="max-h-60 overflow-y-auto space-y-2">
                {uploadedSongs.length === 0 ? (
                  <p className="text-gray-400 text-center py-4">No songs in database yet</p>
                ) : (
                  uploadedSongs.map((song, index) => (
                    <div key={index} className="bg-white/5 rounded-lg p-3 flex items-center justify-between">
                      <div>
                        <p className="font-medium">{song.song_id}</p>
                        <p className="text-sm text-gray-400">{song.features_count} features</p>
                      </div>
                      <button
                        onClick={() => deleteSong(song.song_id)}
                        className="text-red-400 hover:text-red-300 transition-colors"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Analysis Results */}
            {similarityResults.length > 0 && (
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                <h3 className="text-xl font-semibold mb-4">Analysis Results</h3>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {similarityResults.map((result, index) => (
                    <div key={index} className="bg-white/5 rounded-lg p-3">
                      <div className="flex justify-between items-center">
                        <span className="font-medium">{result.song_id}</span>
                        <span className="text-green-400 font-bold">
                          {(result.similarity_score * 100).toFixed(1)}%
                        </span>
                      </div>
                      {result.metadata && Object.keys(result.metadata).length > 0 && (
                        <div className="text-sm text-gray-400 mt-1">
                          {JSON.stringify(result.metadata, null, 2)}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right Column - Real-time Detection */}
          <div className="space-y-6">
            {/* Real-time Detection */}
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
              <h2 className="text-2xl font-semibold mb-4 flex items-center">
                {isRecording ? <Volume2 className="mr-2" size={24} /> : <VolumeX className="mr-2" size={24} />}
                Real-Time Detection
              </h2>
              
              <div className="text-center">
                {!isRecording ? (
                  <button
                    onClick={startRealtimeDetection}
                    disabled={connectionStatus !== 'connected'}
                    className="bg-gradient-to-r from-green-500 to-blue-500 hover:from-green-600 hover:to-blue-600 disabled:from-gray-500 disabled:to-gray-600 text-white py-4 px-8 rounded-full text-lg font-semibold transition-all duration-200 flex items-center justify-center mx-auto"
                  >
                    <Mic className="mr-3" size={24} />
                    Start Listening
                  </button>
                ) : (
                  <button
                    onClick={stopRealtimeDetection}
                    className="bg-gradient-to-r from-red-500 to-pink-500 hover:from-red-600 hover:to-pink-600 text-white py-4 px-8 rounded-full text-lg font-semibold transition-all duration-200 flex items-center justify-center mx-auto"
                  >
                    <MicOff className="mr-3" size={24} />
                    Stop Listening
                  </button>
                )}
                
                {isRecording && (
                  <div className="mt-4">
                    <div className="flex items-center justify-center">
                      <div className="animate-pulse bg-red-500 rounded-full w-3 h-3 mr-2"></div>
                      <span className="text-red-400 font-medium">Recording...</span>
                    </div>
                    <p className="text-gray-400 text-sm mt-2">
                      Play music near your microphone to detect similarities
                    </p>
                  </div>
                )}
                
                {connectionStatus !== 'connected' && (
                  <p className="text-orange-400 text-sm mt-2">
                    Waiting for server connection...
                  </p>
                )}
              </div>
            </div>

            {/* Real-time Matches */}
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20 min-h-96">
              <h3 className="text-xl font-semibold mb-4 flex items-center">
                <Music className="mr-2" size={20} />
                Live Matches
              </h3>
              
              {realtimeMatches.length === 0 ? (
                <div className="text-center py-12">
                  <div className="text-6xl mb-4">🎵</div>
                  <p className="text-gray-400">
                    {isRecording ? 'Listening for music...' : 'Start detection to see matches here'}
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {realtimeMatches.map((match, index) => (
                    <div key={index} className="bg-gradient-to-r from-green-500/20 to-blue-500/20 rounded-lg p-4 border border-green-500/30 animate-pulse">
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-bold text-green-400">🎯 MATCH DETECTED</span>
                        <span className="text-white font-bold text-lg">
                          {(match.similarity * 100).toFixed(1)}%
                        </span>
                      </div>
                      <p className="font-medium text-lg">{match.song_id}</p>
                      {match.metadata && Object.keys(match.metadata).length > 0 && (
                        <div className="text-sm text-gray-300 mt-2">
                          <pre className="whitespace-pre-wrap">
                            {JSON.stringify(match.metadata, null, 2)}
                          </pre>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MusicDetectionApp;