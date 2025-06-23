#!/usr/bin/env python3
"""
Music Detection System Startup Script
Handles system initialization, dependency checks, and provides a CLI interface
"""

import os
import sys
import subprocess
import argparse
from fastapi import FastAPI
from pathlib import Path
import requests
import time

app = FastAPI()

class MusicDetectionSetup:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.required_files = [
            'music_similarity_api.py',
            'music_similarity_engine.py',
            'requirements.txt'
        ]
        
    def check_dependencies(self):
        """Check if all required files and dependencies are present"""
        print("🔍 Checking system dependencies...")
        
        # Check required files
        missing_files = []
        for file in self.required_files:
            if not (self.base_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing required files: {', '.join(missing_files)}")
            return False
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ is required")
            return False
        
        print("✅ All dependencies check passed")
        return True
    
    def install_requirements(self):
        """Install Python requirements"""
        print("📦 Installing Python requirements...")
        
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', 
                str(self.base_dir / 'requirements.txt')
            ])
            print("✅ Requirements installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            return False
    
    def check_audio_system(self):
        """Check if audio system is working"""
        print("🎵 Checking audio system...")
        
        try:
            import sounddevice as sd
            import librosa
            
            # Test audio device access
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            
            if not input_devices:
                print("⚠️  No audio input devices found")
                return False
            
            print(f"✅ Found {len(input_devices)} audio input device(s)")
            return True
            
        except ImportError as e:
            print(f"❌ Audio libraries not properly installed: {e}")
            return False
        except Exception as e:
            print(f"⚠️  Audio system warning: {e}")
            return True  # Continue anyway
    
    def start_backend(self):
        """Start the backend API server"""
        print("🚀 Starting backend server...")
        
        api_file = self.base_dir / 'music_similarity_api.py'
        
        try:
            # Start server in background
            process = subprocess.Popen([
                sys.executable, str(api_file)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Test if server is running
            try:
                response = requests.get('http://localhost:8000/health', timeout=5)
                if response.status_code == 200:
                    print("✅ Backend server started successfully")
                    print("🌐 API available at: http://localhost:8000")
                    print("📚 Documentation at: http://localhost:8000/docs")
                    return process
                else:
                    print(f"❌ Server responded with status: {response.status_code}")
                    process.terminate()
                    return None
            except requests.exceptions.RequestException:
                print("❌ Could not connect to backend server")
                process.terminate()
                return None
                
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return None
    
    def run_cli_mode(self):
        """Run in CLI mode for testing"""
        print("🎵 Starting CLI mode...")
        
        try:
            from music_similarity_engine import MusicDatabase, RealTimeSimilarityDetector
            
            # Initialize system
            db = MusicDatabase()
            detector = RealTimeSimilarityDetector(db)
            
            # Show database stats
            stats = db.get_database_stats()
            print(f"📊 Database: {stats['total_songs']} songs")
            
            def cli_callback(matches):
                print("\n🎯 MATCH DETECTED!")
                for song_id, similarity, metadata in matches[:3]:
                    print(f"  📀 {song_id}: {similarity:.1%}")
            
            print("\n🎤 Starting real-time detection...")
            print("Play music near your microphone or press Ctrl+C to stop")
            
            detector.start_monitoring(similarity_callback=cli_callback)
            
        except ImportError as e:
            print(f"❌ Could not import required modules: {e}")
        except KeyboardInterrupt:
            print("\n👋 Stopped CLI mode")
    
    def add_sample_songs(self):
        """Add sample songs if available"""
        print("🎵 Looking for sample audio files...")
        
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac']
        sample_dir = self.base_dir / 'samples'
        
        if not sample_dir.exists():
            print("📁 Create a 'samples' directory and add audio files to test the system")
            return
        
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(sample_dir.glob(f'*{ext}'))
        
        if not audio_files:
            print("📁 No audio files found in samples directory")
            return
        
        try:
            from music_similarity_engine import MusicDatabase
            
            db = MusicDatabase()
            added_count = 0
            
            for audio_file in audio_files[:5]:  # Limit to 5 files
                song_id = audio_file.stem
                print(f"📀 Adding: {song_id}")
                
                if db.add_song(song_id, str(audio_file), {
                    'filename': audio_file.name,
                    'source': 'sample'
                }):
                    added_count += 1
                    print(f"✅ Added: {song_id}")
                else:
                    print(f"❌ Failed to add: {song_id}")
            
            print(f"📊 Added {added_count} sample songs to database")
            
        except Exception as e:
            print(f"❌ Error adding sample songs: {e}")


def main():
    parser = argparse.ArgumentParser(description='Music Detection System')
    parser.add_argument('--mode', choices=['full', 'api', 'cli'], default='full',
                       help='Run mode: full (API + frontend), api (API only), cli (command line)')
    parser.add_argument('--install', action='store_true',
                       help='Install requirements before starting')
    parser.add_argument('--add-samples', action='store_true',
                       help='Add sample songs from samples/ directory')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip dependency checks')
    
    args = parser.parse_args()
    
    setup = MusicDetectionSetup()
    
    print("🎵 Music Detection System Startup")
    print("=" * 40)
    
    # Install requirements if requested
    if args.install:
        if not setup.install_requirements():
            sys.exit(1)
    
    # Check dependencies
    if not args.skip_checks:
        if not setup.check_dependencies():
            print("\n💡 Try running with --install to install requirements")
            sys.exit(1)
        
        if not setup.check_audio_system():
            print("⚠️  Audio system issues detected, but continuing...")
    
    # Add sample songs if requested
    if args.add_samples:
        setup.add_sample_songs()
    
    # Start based on mode
    if args.mode == 'cli':
        setup.run_cli_mode()
    
    elif args.mode == 'api':
        process = setup.start_backend()
        if process:
            try:
                print("\n📡 Backend server running...")
                print("Press Ctrl+C to stop")
                process.wait()
            except KeyboardInterrupt:
                print("\n👋 Stopping server...")
                process.terminate()
    
    elif args.mode == 'full':
        process = setup.start_backend()
        if process:
            try:
                print("\n🌐 Full system running!")
                print("📱 Open http://localhost:8000/docs for API documentation")
                print("💻 Use the React frontend for the web interface")
                print("Press Ctrl+C to stop")
                process.wait()
            except KeyboardInterrupt:
                print("\n👋 Stopping system...")
                process.terminate()
        else:
            print("❌ Failed to start backend, trying CLI mode...")
            setup.run_cli_mode()


if __name__ == "__main__":
    main()