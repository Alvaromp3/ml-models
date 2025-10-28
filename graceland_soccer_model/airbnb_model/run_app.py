#!/usr/bin/env python3
import subprocess
import sys
import os

def main():
    print("🚀 Iniciando aplicación web de predicción de precios Airbnb...")
    print("📊 Entrenando modelo...")
    
    # Entrenar el modelo primero
    try:
        result = subprocess.run([sys.executable, "modelo_airbnb.py"], 
                              capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print("✅ Modelo entrenado exitosamente")
        else:
            print(f"❌ Error entrenando modelo: {result.stderr}")
            return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    port = sys.argv[1] if len(sys.argv) > 1 else "5000"
    print("🌐 Iniciando servidor web...")
    print(f"📱 Abre tu navegador en: http://localhost:{port}")
    print("⏹️  Presiona Ctrl+C para detener el servidor")
    
    # Ejecutar la aplicación Flask
    try:
        subprocess.run([sys.executable, "app.py", port])
    except KeyboardInterrupt:
        print("\n👋 Servidor detenido")

if __name__ == "__main__":
    main()