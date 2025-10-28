#!/usr/bin/env python3

import subprocess
import sys
import os

def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        print("Iniciando FIFA Score Predictor...")
        print("La aplicación se abrirá en http://localhost:8501")
        print("="*50)
        
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
        
    except KeyboardInterrupt:
        print("\nFIFA Score Predictor detenido")
    except Exception as e:
        print(f"Error: {e}")
        print("Asegúrate de tener instaladas las dependencias: pip install -r requirements.txt")

if __name__ == "__main__":
    main()