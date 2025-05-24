import requests
from flask import Flask, request
from flask_socketio import SocketIO, emit
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de Flask y SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuración de la API externa
EXTERNAL_API_URL = os.getenv("EXTERNAL_API_URL", "https://api.openai.com/v1/chat/completions")
API_KEY = os.getenv("API_KEY")  # Asegúrate de tener esta variable en tu .env

# Headers para la solicitud a la API externa
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def query_external_model(prompt):
    """Envía una solicitud a la API externa y devuelve la respuesta"""
    try:
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        response = requests.post(EXTERNAL_API_URL, headers=HEADERS, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    except Exception as e:
        print(f"Error al consultar la API externa: {e}")
        return "Lo siento, ocurrió un error al procesar tu mensaje."

# Manejo de conexiones SocketIO
@socketio.on('connect')
def handle_connect():
    print('Cliente conectado:', request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    print('Cliente desconectado:', request.sid)

@socketio.on('message')
def handle_message(data):
    print('Mensaje recibido desde el cliente:', data)
    if isinstance(data, dict) and 'text' in data:
        user_input = data['text']
        
        # Obtener respuesta del modelo externo
        response = query_external_model(user_input)
        
        emit('response', {'text': response})
    else:
        emit('error', {'text': 'Formato de mensaje inválido'})

# Función principal
def main():
    # Iniciar el servidor SocketIO
    print("Servidor SocketIO iniciado. Esperando conexiones...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    main()