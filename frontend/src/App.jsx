import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, 
  Music, 
  Play, 
  Pause, 
  Mic, 
  MicOff, 
  Database, 
  Search, 
  Trash2, 
  Volume2,
  Activity,
  FileAudio,
  AlertCircle,
  CheckCircle,
  Loader
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const App = () => {
  // State management
  const [activeTab, setActiveTab] = useState('upload');
  const [songs, setSongs] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [similarityResults, setSimilarityResults] = useState([]);
  const [isRealTimeActive, setIsRealTimeActive] = useState(false);
  const [realtimeMatches, setRealtimeMatches] = useState([]);
  const [notification, setNotification] = useState(null);
  
  // Refs
  const fileInputRef = useRef(null);
  const analyzeInputRef = useRef(null);
  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Fetch songs from database
  const fetchSongs = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/database/songs`);
      const data = await response.json();
      setSongs(data.songs || []);
    } catch (error) {
      showNotification('Error fetching songs', 'error');
    }
  };

  // Show notification
  const showNotification = (message, type = 'info') => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 3000);
  };

  // Initialize component
  useEffect(() => {
    fetchSongs();
  }, []);

  // Upload song to database
  const handleUploadSong = async (file, metadata = {}) => {
    setIsLoading(true);
    setUploadProgress(0);
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('song_id', file.name.split('.')[0]);
    formData.append('metadata', JSON.stringify(metadata));

    try {
      const response = await fetch(`${API_BASE_URL}/upload-song`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        showNotification(`Song "${result.song_id}" uploaded successfully!`, 'success');
        fetchSongs();
        setUploadProgress(100);
      } else {
        throw new Error('Upload failed');
      }
    } catch (error) {
      showNotification('Error uploading song', 'error');
    } finally {
      setIsLoading(false);
      setTimeout(() => setUploadProgress(0), 1000);
    }
  };

  // Analyze audio file
  const handleAnalyzeAudio = async (file) => {
    setIsLoading(true);
    setSimilarityResults([]);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/analyze-audio`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        setSimilarityResults(result.similar_songs || []);
        showNotification(`Found ${result.total_matches} similar songs!`, 'success');
      } else {
        throw new Error('Analysis failed');
      }
    } catch (error) {
      showNotification('Error analyzing audio', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  // Delete song from database
  const handleDeleteSong = async (songId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/database/songs/${songId}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        showNotification(`Song "${songId}" deleted successfully!`, 'success');
        fetchSongs();
      } else {
        throw new Error('Delete failed');
      }
    } catch (error) {
      showNotification('Error deleting song', 'error');
    }
  };

  // Start real-time detection
  const startRealTimeDetection = async () => {
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Create WebSocket connection
      wsRef.current = new WebSocket(`ws://localhost:8000/ws/realtime`);
      
      wsRef.current.onopen = () => {
        setIsRealTimeActive(true);
        setRealtimeMatches([]);
        showNotification('Real-time detection started!', 'success');
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'similarity_detected') {
          setRealtimeMatches(data.matches);
        }
      };

      wsRef.current.onclose = () => {
        setIsRealTimeActive(false);
        showNotification('Real-time detection stopped', 'info');
      };

      // Setup audio recording
      mediaRecorderRef.current = new MediaRecorder(stream);
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
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
        
        reader.readAsDataURL(audioBlob);
        audioChunksRef.current = [];
      };

      // Start recording in chunks
      mediaRecorderRef.current.start();
      const recordingInterval = setInterval(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
          mediaRecorderRef.current.stop();
          mediaRecorderRef.current.start();
        }
      }, 2000);

      // Store interval for cleanup
      wsRef.current.recordingInterval = recordingInterval;

    } catch (error) {
      showNotification('Error starting real-time detection', 'error');
    }
  };

  // Stop real-time detection
  const stopRealTimeDetection = () => {
    if (wsRef.current) {
      clearInterval(wsRef.current.recordingInterval);
      wsRef.current.close();
    }
    
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
    
    setIsRealTimeActive(false);
    setRealtimeMatches([]);
  };

  // File upload component
  const FileUploadZone = ({ onFileSelect, accept, children }) => {
    const [isDragging, setIsDragging] = useState(false);

    const handleDragOver = (e) => {
      e.preventDefault();
      setIsDragging(true);
    };

    const handleDragLeave = (e) => {
      e.preventDefault();
      setIsDragging(false);
    };

    const handleDrop = (e) => {
      e.preventDefault();
      setIsDragging(false);
      
      const files = Array.from(e.dataTransfer.files);
      if (files.length > 0 && files[0].type.startsWith('audio/')) {
        onFileSelect(files[0]);
      }
    };

    return (
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200 ${
          isDragging 
            ? 'border-blue-500 bg-blue-50' 
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => document.getElementById('file-input').click()}
      >
        {children}
        <input
          id="file-input"
          type="file"
          accept={accept}
          className="hidden"
          onChange={(e) => e.target.files[0] && onFileSelect(e.target.files[0])}
        />
      </div>
    );
  };

  // Similarity result card
  const SimilarityCard = ({ result }) => (
    <div className="bg-white rounded-lg shadow-md p-4 border-l-4 border-blue-500">
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-semibold text-lg">{result.song_id}</h3>
        <div className="flex items-center space-x-2">
          <div className="text-sm text-gray-600">
            {(result.similarity_score * 100).toFixed(1)}% match
          </div>
          <div className={`w-3 h-3 rounded-full ${
            result.similarity_score > 0.9 ? 'bg-green-500' : 
            result.similarity_score > 0.8 ? 'bg-yellow-500' : 'bg-red-500'
          }`} />
        </div>
      </div>
      
      {result.metadata && (
        <div className="text-sm text-gray-600 space-y-1">
          {result.metadata.artist && <div>Artist: {result.metadata.artist}</div>}
          {result.metadata.genre && <div>Genre: {result.metadata.genre}</div>}
          {result.metadata.year && <div>Year: {result.metadata.year}</div>}
        </div>
      )}
      
      <div className="mt-3">
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-blue-500 h-2 rounded-full transition-all duration-300"
            style={{ width: `${result.similarity_score * 100}%` }}
          />
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50">
      {/* Notification */}
      {notification && (
        <div className={`fixed top-4 right-4 z-50 px-4 py-2 rounded-lg shadow-lg flex items-center space-x-2 ${
          notification.type === 'success' ? 'bg-green-500 text-white' :
          notification.type === 'error' ? 'bg-red-500 text-white' :
          'bg-blue-500 text-white'
        }`}>
          {notification.type === 'success' && <CheckCircle size={20} />}
          {notification.type === 'error' && <AlertCircle size={20} />}
          <span>{notification.message}</span>
        </div>
      )}

      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <Music className="h-8 w-8 text-purple-600" />
              <h1 className="text-xl font-bold text-gray-900">Music Similarity Detection</h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-600">
                {songs.length} songs in database
              </div>
              <div className={`flex items-center space-x-2 ${isRealTimeActive ? 'text-green-600' : 'text-gray-400'}`}>
                <Activity size={16} />
                <span className="text-sm">Real-time</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tab Navigation */}
        <div className="mb-8">
          <nav className="flex space-x-8">
            {[
              { id: 'upload', label: 'Upload Songs', icon: Upload },
              { id: 'analyze', label: 'Analyze Audio', icon: Search },
              { id: 'realtime', label: 'Real-time Detection', icon: Mic },
              { id: 'database', label: 'Database', icon: Database },
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                  activeTab === id
                    ? 'bg-purple-600 text-white'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                <Icon size={20} />
                <span>{label}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          {/* Upload Tab */}
          {activeTab === 'upload' && (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-gray-900">Upload Songs to Database</h2>
              
              <FileUploadZone 
                onFileSelect={(file) => handleUploadSong(file)}
                accept="audio/*"
              >
                <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                <p className="text-lg text-gray-600 mb-2">Drop audio files here or click to browse</p>
                <p className="text-sm text-gray-500">Supports MP3, WAV, FLAC, and other audio formats</p>
              </FileUploadZone>

              {uploadProgress > 0 && (
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              )}

              {isLoading && (
                <div className="flex items-center justify-center py-4">
                  <Loader className="animate-spin h-6 w-6 text-purple-600 mr-2" />
                  <span>Processing audio file...</span>
                </div>
              )}
            </div>
          )}

          {/* Analyze Tab */}
          {activeTab === 'analyze' && (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-gray-900">Analyze Audio for Similarities</h2>
              
              <FileUploadZone 
                onFileSelect={(file) => handleAnalyzeAudio(file)}
                accept="audio/*"
              >
                <Search className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                <p className="text-lg text-gray-600 mb-2">Upload audio to find similar songs</p>
                <p className="text-sm text-gray-500">Compare against {songs.length} songs in database</p>
              </FileUploadZone>

              {isLoading && (
                <div className="flex items-center justify-center py-8">
                  <Loader className="animate-spin h-8 w-8 text-purple-600 mr-3" />
                  <span className="text-lg">Analyzing audio features...</span>
                </div>
              )}

              {similarityResults.length > 0 && (
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Similarity Results</h3>
                  <div className="grid gap-4 md:grid-cols-2">
                    {similarityResults.map((result, index) => (
                      <SimilarityCard key={index} result={result} />
                    ))}
                  </div>
                </div>
              )}

              {similarityResults.length === 0 && !isLoading && (
                <div className="text-center py-8 text-gray-500">
                  <FileAudio className="mx-auto h-12 w-12 mb-4" />
                  <p>No similar songs found. Try uploading an audio file to analyze.</p>
                </div>
              )}
            </div>
          )}

          {/* Real-time Tab */}
          {activeTab === 'realtime' && (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-gray-900">Real-time Similarity Detection</h2>
              
              <div className="text-center">
                <button
                  onClick={isRealTimeActive ? stopRealTimeDetection : startRealTimeDetection}
                  className={`inline-flex items-center space-x-3 px-8 py-4 rounded-lg text-lg font-semibold transition-colors ${
                    isRealTimeActive
                      ? 'bg-red-600 hover:bg-red-700 text-white'
                      : 'bg-green-600 hover:bg-green-700 text-white'
                  }`}
                >
                  {isRealTimeActive ? <MicOff size={24} /> : <Mic size={24} />}
                  <span>{isRealTimeActive ? 'Stop Detection' : 'Start Detection'}</span>
                </button>
              </div>

              {isRealTimeActive && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-4 text-center">
                  <div className="flex items-center justify-center space-x-2 mb-2">
                    <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
                    <span className="text-green-800 font-medium">Listening for audio...</span>
                  </div>
                  <p className="text-sm text-green-600">Play some music near your microphone</p>
                </div>
              )}

              {realtimeMatches.length > 0 && (
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold flex items-center space-x-2">
                    <Volume2 className="text-green-600" />
                    <span>Live Matches</span>
                  </h3>
                  <div className="grid gap-4 md:grid-cols-2">
                    {realtimeMatches.map((match, index) => (
                      <SimilarityCard key={index} result={match} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Database Tab */}
          {activeTab === 'database' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold text-gray-900">Music Database</h2>
                <button
                  onClick={fetchSongs}
                  className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  <Database size={20} />
                  <span>Refresh</span>
                </button>
              </div>

              {songs.length === 0 ? (
                <div className="text-center py-12 text-gray-500">
                  <Database className="mx-auto h-16 w-16 mb-4" />
                  <p className="text-lg">No songs in database</p>
                  <p className="text-sm">Upload some songs to get started</p>
                </div>
              ) : (
                <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 rounded-lg">
                  <table className="min-w-full divide-y divide-gray-300">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Song ID
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Artist
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Genre
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Features
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Actions
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {songs.map((song) => (
                        <tr key={song.song_id} className="hover:bg-gray-50">
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                            {song.song_id}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {song.metadata?.artist || 'Unknown'}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {song.metadata?.genre || 'Unknown'}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {song.features_count} chunks
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            <button
                              onClick={() => handleDeleteSong(song.song_id)}
                              className="text-red-600 hover:text-red-900 transition-colors"
                            >
                              <Trash2 size={16} />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default App;
