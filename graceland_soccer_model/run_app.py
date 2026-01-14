#!/usr/bin/env python3
"""
Run script for Elite Sports Performance Analytics application
"""
import subprocess
import sys
import os

def main():
    """Main entry point for running the Streamlit app"""
    
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit is not installed!")
        print("Please install dependencies first:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_file = os.path.join(script_dir, 'app.py')
    
    if not os.path.exists(app_file):
        print(f"❌ app.py not found in {script_dir}")
        sys.exit(1)
    
    print("🚀 Starting Elite Sports Performance Analytics...")
    print("📊 Application will open in your browser")
    print("📍 URL: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_file,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.maxUploadSize", "200",
            "--server.maxMessageSize", "200"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

