import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import Dataset
import torch
import os
from flask import Flask, request
from flask_socketio import SocketIO, emit
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
import time
import re
from typing import Dict, List, Optional
import gc

# Configurar logging detallado
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración de Flask y SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Variables globales para el modelo
model = None
tokenizer = None
embedding_model = None
chat_pipeline = None

# Configurar ChromaDB
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="documentos_entrenamiento",
        metadata={"hnsw:space": "cosine"}
    )
    logger.info("✅ ChromaDB configurado correctamente")
except Exception as e:
    logger.error(f"❌ Error inicializando ChromaDB: {e}")
    client = chromadb.Client()
    collection = client.create_collection(name="documentos_entrenamiento")

# Datos del restaurante en español (mantener igual)
documents = [
    # Disponibilidad de Mesas
    "Tenemos disponibilidad para 4 personas a las 8:00 PM esta noche",
    "Lamentablemente estamos completos para esta noche, ¿le gustaría reservar para otro día?",
    "Disponemos de mesa para 2 personas a las 9:30 PM",
    "Tenemos disponibilidad en nuestra terraza para 6 personas",
    
    # Menú y Especialidades
    "Nuestro especial del día es risotto de hongos silvestres con trufa",
    "Tenemos varias opciones vegetarianas: lasaña de berenjena, curry de verduras y ensalada mediterránea",
    "El plato estrella de la casa es el solomillo ibérico con foie",
    "Hoy tenemos como especialidad paella marinera para dos personas",
    "El menú del día incluye primer plato, segundo plato, postre y bebida por 15€",
    "Los martes tenemos promoción de tapas con cerveza a 5€",
    
    # Reservas
    "Su reserva para 2 personas a las 8:00 PM ha sido confirmada. Código: RES12345",
    "¿En qué fecha y hora le gustaría modificar su reserva?",
    "Necesito su nombre y número de teléfono para completar la reserva",
    "Su reserva ha sido cancelada exitosamente",
    
    # Horarios
    "Nuestro horario es de lunes a sábado de 12:00 PM a 11:00 PM, domingos de 12:00 PM a 10:00 PM",
    "Los viernes y sábados extendemos horario hasta medianoche",
    "El horario de cocina cierra 30 minutos antes del cierre del restaurante",
    
    # Ubicación y Contacto
    "Estamos ubicados en Calle Principal 123, Ciudad. Teléfono: +34 912 345 678",
    "Puede encontrarnos en el centro comercial Plaza Mayor, planta baja",
    "Para contactarnos directamente: teléfono +34 912 345 678 o email info@restaurante.com",
    
    # Servicios Adicionales
    "Sí, ofrecemos servicio a domicilio con un mínimo de pedido de 15€",
    "Aceptamos todas las tarjetas de crédito principales y también efectivo",
    "Tenemos servicio de catering para eventos y celebraciones",
    "Ofrecemos menú degustación con reserva previa",
    
    # Cortesía y Servicio
    "¡Bienvenido! ¿En qué puedo ayudarle hoy?",
    "Gracias por contactarnos. ¿Hay algo más en lo que pueda asistirle?",
    "Ha sido un placer atenderle, ¡esperamos verle pronto!",
    "¿Necesita ayuda con algo más?"
]

metadatas = [
    # Disponibilidad
    {"id": "disp_mesa_001", "categoria": "disponibilidad", "tipo": "confirmacion", "capacidad": "4"},
    {"id": "disp_mesa_002", "categoria": "disponibilidad", "tipo": "negacion", "capacidad": "0"},
    {"id": "disp_mesa_003", "categoria": "disponibilidad", "tipo": "confirmacion", "capacidad": "2"},
    {"id": "disp_mesa_004", "categoria": "disponibilidad", "tipo": "confirmacion", "capacidad": "6"},
    
    # Menú
    {"id": "menu_especial_001", "categoria": "menu", "tipo": "especialidad", "plato": "risotto"},
    {"id": "menu_vegetariano_001", "categoria": "menu", "tipo": "opciones_especiales", "dieta": "vegetariana"},
    {"id": "menu_estrella_001", "categoria": "menu", "tipo": "especialidad", "plato": "solomillo_iberico"},
    {"id": "menu_especial_002", "categoria": "menu", "tipo": "especialidad", "plato": "paella"},
    {"id": "menu_dia_001", "categoria": "menu", "tipo": "menu_dia", "precio": "15"},
    {"id": "menu_promocion_001", "categoria": "menu", "tipo": "promocion", "dia": "martes"},
    
    # Reservas
    {"id": "reserva_confirm_001", "categoria": "reserva", "tipo": "confirmacion", "personas": "2"},
    {"id": "reserva_modif_001", "categoria": "reserva", "tipo": "modificacion", "accion": "modificar"},
    {"id": "reserva_info_001", "categoria": "reserva", "tipo": "solicitud_info", "dato": "contacto"},
    {"id": "reserva_cancel_001", "categoria": "reserva", "tipo": "cancelacion", "estado": "cancelada"},
    
    # Horarios
    {"id": "horario_001", "categoria": "horario", "tipo": "informacion_general", "dias": "todos"},
    {"id": "horario_002", "categoria": "horario", "tipo": "extension", "dias": "fin_semana"},
    {"id": "horario_003", "categoria": "horario", "tipo": "informacion_especifica", "servicio": "cocina"},
    
    # Ubicación
    {"id": "ubicacion_001", "categoria": "ubicacion", "tipo": "direccion_contacto", "info": "completa"},
    {"id": "ubicacion_002", "categoria": "ubicacion", "tipo": "direccion", "lugar": "centro_comercial"},
    {"id": "contacto_001", "categoria": "contacto", "tipo": "informacion_contacto", "medios": "multiple"},
    
    # Servicios
    {"id": "servicio_001", "categoria": "servicios", "tipo": "domicilio", "minimo": "15"},
    {"id": "servicio_002", "categoria": "servicios", "tipo": "metodos_pago", "medios": "multiple"},
    {"id": "servicio_003", "categoria": "servicios", "tipo": "catering", "eventos": "true"},
    {"id": "servicio_004", "categoria": "servicios", "tipo": "menu_especial", "reserva": "requerida"},
    
    # Cortesía
    {"id": "cortesia_001", "categoria": "cortesia", "tipo": "saludo", "tono": "amable"},
    {"id": "cortesia_002", "categoria": "cortesia", "tipo": "despedida", "tono": "agradecido"},
    {"id": "cortesia_003", "categoria": "cortesia", "tipo": "despedida", "tono": "cordial"},
    {"id": "cortesia_004", "categoria": "cortesia", "tipo": "oferta_ayuda", "tono": "servicial"}
]

ids = [
    "disp_mesa_001", "disp_mesa_002", "disp_mesa_003", "disp_mesa_004",
    "menu_especial_001", "menu_vegetariano_001", "menu_estrella_001", "menu_especial_002",
    "menu_dia_001", "menu_promocion_001",
    "reserva_confirm_001", "reserva_modif_001", "reserva_info_001", "reserva_cancel_001",
    "horario_001", "horario_002", "horario_003",
    "ubicacion_001", "ubicacion_002", "contacto_001",
    "servicio_001", "servicio_002", "servicio_003", "servicio_004",
    "cortesia_001", "cortesia_002", "cortesia_003", "cortesia_004"
]

# Función para inicializar datos en ChromaDB
def inicializar_chroma():
    try:
        if collection.count() == 0:
            logger.info("Inicializando datos en ChromaDB...")
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"✅ Datos cargados: {collection.count()} documentos")
        else:
            logger.info(f"📊 Colección ya contiene {collection.count()} documentos")
    except Exception as e:
        logger.error(f"❌ Error inicializando ChromaDB: {e}")

# Función mejorada de búsqueda en ChromaDB
def buscar_chroma(user_input: str, top_k: int = 5, threshold: float = 0.4) -> Optional[str]:
    try:
        logger.info(f"🔍 Buscando en ChromaDB: '{user_input}'")
        
        resultados = collection.query(
            query_texts=[user_input],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        if resultados['documents'] and len(resultados['documents'][0]) > 0:
            # Buscar el documento más relevante
            for i, distancia in enumerate(resultados['distances'][0]):
                if distancia < threshold:
                    contexto = resultados['documents'][0][i]
                    logger.info(f"✅ Contexto encontrado: '{contexto}'")
                    return contexto
            
            # Si no hay coincidencia fuerte, devolver el más cercano
            contexto = resultados['documents'][0][0]
            logger.info(f"⚠️  Usando contexto más cercano: '{contexto}'")
            return contexto
        
        logger.info("ℹ️  No se encontró contexto relevante en ChromaDB")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error en búsqueda Chroma: {e}")
        return None

# Cargar modelo liviano pero efectivo
def cargar_modelo_liviano():
    logger.info("🔄 Cargando modelo liviano optimizado...")
    try:
        # Opción 1: Microsoft's DialoGPT-medium (más estable)
        model_name = "microsoft/DialoGPT-medium"
        
        # Opción 2: GPT2 en español (más liviano)
        # model_name = "datificate/gpt2-small-spanish"
        
        # Opción 3: DistilGPT2 (el más liviano)
        # model_name = "distilgpt2"
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Cargar modelo con configuración optimizada
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Usar float32 para mayor estabilidad
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Configurar modelo para inferencia
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()
        
        logger.info(f"✅ Modelo {model_name} cargado correctamente")
        return model, tokenizer, None, None
        
    except Exception as e:
        logger.error(f"❌ Error cargando modelo liviano: {e}")
        # Fallback extremo - solo ChromaDB + respuestas predefinidas
        logger.warning("⚠️  Usando modo solo ChromaDB + reglas")
        return None, None, None, None

# Sistema de templates optimizado
def crear_prompt_optimizado(user_input: str, contexto_chroma: Optional[str] = None) -> str:
    """
    Crea un prompt optimizado y seguro
    """
    sistema = "Eres un asistente de restaurante español. Responde de manera amable y profesional."
    
    contexto = contexto_chroma if contexto_chroma else "Información general del restaurante."
    
    prompt = f"{sistema}\nContexto: {contexto}\nPregunta: {user_input}\nRespuesta:"
    
    return prompt

# Generar respuesta segura y controlada
def generar_respuesta_segura(prompt: str, max_tokens: int = 100) -> str:
    global model, tokenizer
    
    try:
        if model is None or tokenizer is None:
            raise ValueError("Modelo no disponible")
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
            return_attention_mask=True
        )

        # Mover a dispositivo adecuado
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2,
                early_stopping=True
            )

        respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraer solo la parte de la respuesta
        respuesta = respuesta.replace(prompt, "").strip()
        
        # Limpiar respuesta
        respuesta = re.sub(r'<\|.*?\|>', '', respuesta)  # Remover tokens especiales
        respuesta = respuesta.split('\n')[0].strip()  # Tomar primera línea
        
        return respuesta if respuesta else "¿En qué más puedo ayudarte?"
            
    except Exception as e:
        logger.error(f"❌ Error en generación segura: {e}")
        raise e

# Sistema de respuestas con múltiples capas de fallback
def generar_respuesta_inteligente(user_input: str, contexto_chroma: Optional[str] = None) -> str:
    """Sistema de respuestas con fallback escalonado"""
    
    # Capa 1: Respuestas predefinidas específicas
    respuesta_predefinida = obtener_respuesta_predefinida(user_input, contexto_chroma)
    if respuesta_predefinida:
        return respuesta_predefinida
    
    # Capa 2: Usar contexto de ChromaDB
    if contexto_chroma:
        return contexto_chroma
    
    # Capa 3: Modelo AI (si está disponible)
    try:
        prompt = crear_prompt_optimizado(user_input, contexto_chroma)
        respuesta = generar_respuesta_segura(prompt)
        return respuesta
    except Exception as e:
        logger.warning(f"⚠️  Modelo AI no disponible: {e}")
    
    # Capa 4: Respuesta genérica
    return "¿En qué más puedo ayudarte? Puedo ayudarte con información sobre menú, reservas, horarios y ubicación."

# Respuestas predefinidas mejoradas
def obtener_respuesta_predefinida(user_input: str, contexto_chroma: Optional[str] = None) -> Optional[str]:
    user_input_lower = user_input.lower()
    
    # Palabras clave y respuestas
    respuestas = {
        'hola': "¡Hola! Bienvenido/a al restaurante. ¿En qué puedo ayudarte hoy?",
        'buenas': "¡Buenas! ¿Cómo puedo ayudarte?",
        'gracias': "¡Gracias a ti! ¿Necesitas ayuda con algo más?",
        'adios': "¡Hasta luego! Esperamos verte pronto.",
        'menu': "Nuestro menú incluye especialidades como risotto de hongos, solomillo ibérico y opciones vegetarianas.",
        'reserva': "Para reservas, necesito saber para cuántas personas, a qué hora y para qué fecha.",
        'mesa': "¿Para cuántas personas y a qué hora te gustaría la mesa?",
        'horario': "Nuestro horario es de lunes a sábado de 12:00 PM a 11:00 PM, domingos de 12:00 PM a 10:00 PM",
        'ubicacion': "Estamos ubicados en Calle Principal 123, Ciudad. Teléfono: +34 912 345 678",
        'precio': "El menú del día cuesta 15€. Los platos principales varían entre 12€ y 25€.",
        'vegetariano': "Tenemos opciones vegetarianas: lasaña de berenjena, curry de verduras y ensalada mediterránea."
    }
    
    # Buscar coincidencia exacta primero
    for keyword, respuesta in respuestas.items():
        if keyword in user_input_lower:
            return respuesta
    
    # Usar contexto de ChromaDB si está disponible
    if contexto_chroma:
        return contexto_chroma
    
    return None

# FUNCIÓN PRINCIPAL GENERATE_RESPONSE
def generate_response(user_input: str, context: Optional[Dict] = None) -> str:
    global model, tokenizer
    
    try:
        logger.info(f"🧠 Generando respuesta para: '{user_input}'")
        start_time = time.time()
        
        # 1. Buscar contexto relevante
        contexto_chroma = buscar_chroma(user_input)
        
        # 2. Generar respuesta inteligente
        respuesta = generar_respuesta_inteligente(user_input, contexto_chroma)
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Respuesta generada en {processing_time:.2f}s: '{respuesta}'")
        return respuesta
        
    except Exception as e:
        logger.error(f"❌ Error crítico en generate_response: {e}")
        return "Disculpa las molestias. ¿Podrías repetir tu pregunta? Estoy aquí para ayudarte."

# Manejo de conexiones SocketIO
@socketio.on('connect')
def handle_connect():
    logger.info(f'✅ Cliente conectado: {request.sid}')
    emit('status', {'message': 'Conectado al servidor de IA'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f'❌ Cliente desconectado: {request.sid}')

@socketio.on('message')
def handle_message(data):
    try:
        logger.info(f'📥 Mensaje recibido: {data}')
        
        if isinstance(data, dict) and 'text' in data:
            user_input = data['text']
            context = data.get('context', {})
            
            logger.info(f'👤 Mensaje del usuario: "{user_input}"')
            
            # Generar respuesta
            response = generate_response(user_input, context)
            
            # Enviar respuesta
            response_data = {
                'text': response,
                'context': context
            }
            
            logger.info(f'📤 Enviando respuesta: {response_data}')
            emit('response', response_data)
            
        else:
            error_msg = 'Formato de mensaje inválido'
            logger.warning(f'⚠️  {error_msg}: {data}')
            emit('error', {'text': error_msg})
            
    except Exception as e:
        error_msg = f'Error procesando mensaje: {str(e)}'
        logger.error(f'❌ {error_msg}')
        emit('error', {'text': 'Error interno del servidor'})

# Ruta de salud
@app.route('/health')
def health_check():
    return {'status': 'ok', 'model_loaded': model is not None}

# Función principal optimizada
def main():
    global model, tokenizer, embedding_model
    
    try:
        logger.info("🚀 Iniciando servidor de IA optimizado...")
        
        # Configuración optimizada
        torch.set_num_threads(min(2, os.cpu_count() or 1))
        
        # Limpiar memoria
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Inicializar ChromaDB
        inicializar_chroma()
        
        # Cargar el modelo liviano
        model, tokenizer, _, _ = cargar_modelo_liviano()
        
        logger.info("✅ Servidor configurado correctamente")
       
        # Iniciar el servidor
        logger.info("🌐 Servidor SocketIO iniciado. Esperando conexiones...")
        socketio.run(
            app, 
            host='0.0.0.0', 
            port=5000, 
            debug=False,
            use_reloader=False,
            allow_unsafe_werkzeug=True
        )
        
    except Exception as e:
        logger.error(f"💥 Error en la función principal: {e}")
        raise e

if __name__ == "__main__":
    main()