import subprocess
import sys
import os

def run_streamlit_app():
    """Run the Streamlit app"""
    app_path = os.path.join(os.path.dirname(__file__), 'app.py')
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', app_path,
            '--server.port', '8501',
            '--server.address', 'localhost'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
    except KeyboardInterrupt:
        print("\nApp stopped by user")

if __name__ == "__main__":
    run_streamlit_app()