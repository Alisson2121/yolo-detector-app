import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from werkzeug.utils import secure_filename
import base64
from collections import Counter

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Crear carpeta de uploads si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Cargar modelo YOLO
print("Cargando modelo YOLO...")
model = YOLO('yolov8n.pt')  # modelo nano, más rápido
print("Modelo YOLO cargado correctamente")

# Configuración de base de datos - REEMPLAZA CON TUS CREDENCIALES
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:[YOUR-PASSWORD]@db.xkmbefvvasnnqjapvjdt.supabase.co:5432/postgres')

def get_db_connection():
    """Conexión a PostgreSQL"""
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def calcular_forma(x, y, w, h, contorno=None):
    """Determina si es cuadrado, rectángulo o círculo"""
    ratio = w / h if h > 0 else 1
    
    # Cuadrado vs Rectángulo (±10%)
    if 0.9 <= ratio <= 1.1:
        forma = "cuadrado"
    else:
        # Si tenemos contorno, verificar circularidad
        if contorno is not None and len(contorno) >= 5:
            area = cv2.contourArea(contorno)
            perimetro = cv2.arcLength(contorno, True)
            if perimetro > 0:
                circularidad = (4 * np.pi * area) / (perimetro ** 2)
                if circularidad >= 0.85:
                    forma = "circulo"
                else:
                    forma = "rectangulo"
            else:
                forma = "rectangulo"
        else:
            forma = "rectangulo"
    
    return forma

def calcular_color_dominante(imagen, x, y, w, h):
    """Calcula el color dominante en el bounding box"""
    # Extraer región de interés
    roi = imagen[y:y+h, x:x+w]
    
    # Convertir a RGB
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # Reshape para k-means
    pixels = roi_rgb.reshape(-1, 3)
    
    # Usar los píxeles más comunes (más rápido que k-means)
    pixels_list = [tuple(p) for p in pixels]
    color_counts = Counter(pixels_list)
    color_dominante_rgb = color_counts.most_common(1)[0][0]
    
    # Convertir a nombre aproximado
    r, g, b = color_dominante_rgb
    color_nombre = rgb_a_nombre(r, g, b)
    
    return color_nombre

def rgb_a_nombre(r, g, b):
    """Convierte RGB a nombre de color aproximado"""
    colores = {
        'rojo': (255, 0, 0),
        'verde': (0, 255, 0),
        'azul': (0, 0, 255),
        'amarillo': (255, 255, 0),
        'naranja': (255, 165, 0),
        'morado': (128, 0, 128),
        'rosa': (255, 192, 203),
        'negro': (0, 0, 0),
        'blanco': (255, 255, 255),
        'gris': (128, 128, 128),
        'cafe': (165, 42, 42),
        'celeste': (135, 206, 235)
    }
    
    min_dist = float('inf')
    color_cercano = 'desconocido'
    
    for nombre, (cr, cg, cb) in colores.items():
        dist = np.sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
        if dist < min_dist:
            min_dist = dist
            color_cercano = nombre
    
    return color_cercano

@app.route('/')
def index():
    """Página principal"""
    print("GET / - Página principal cargada")
    return render_template('index.html')

@app.route('/detectar', methods=['POST'])
def detectar():
    """Procesa imagen y detecta objeto"""
    print("POST /detectar - Iniciando detección")
    
    try:
        # Obtener imagen
        if 'imagen' not in request.files:
            return jsonify({'error': 'No se envió imagen'}), 400
        
        file = request.files['imagen']
        if file.filename == '':
            return jsonify({'error': 'Archivo vacío'}), 400
        
        # Guardar imagen
        filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"Imagen guardada: {filepath}")
        
        # Leer imagen
        imagen = cv2.imread(filepath)
        
        # Detección YOLO
        print("Ejecutando YOLO...")
        resultados = model(imagen)
        
        if len(resultados[0].boxes) == 0:
            print("No se detectaron objetos")
            return jsonify({'error': 'No se detectaron objetos'}), 400
        
        # Tomar primer objeto detectado
        box = resultados[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        clase_id = int(box.cls[0])
        objeto = model.names[clase_id]
        
        print(f"Objeto detectado: {objeto}")
        
        # Calcular dimensiones
        w = x2 - x1
        h = y2 - y1
        
        # Extraer contorno para forma
        roi_gray = cv2.cvtColor(imagen[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(roi_gray, 127, 255, cv2.THRESH_BINARY)
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contorno = contornos[0] if contornos else None
        
        # Calcular forma y color
        forma = calcular_forma(x1, y1, w, h, contorno)
        color = calcular_color_dominante(imagen, x1, y1, w, h)
        
        print(f"Forma: {forma}, Color: {color}")
        
        # Guardar en base de datos
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO detecciones (objeto, forma, color, fecha_hora, imagen_url) 
               VALUES (%s, %s, %s, %s, %s)""",
            (objeto, forma, color, datetime.now(), f"/static/uploads/{filename}")
        )
        conn.commit()
        cur.close()
        conn.close()
        
        print("Guardado en base de datos correctamente")
        
        return jsonify({
            'objeto': objeto,
            'forma': forma,
            'color': color,
            'imagen_url': f"/static/uploads/{filename}",
            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        })
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ultimo', methods=['GET'])
def ultimo():
    """Devuelve el último registro"""
    print("GET /api/ultimo - Consultando último registro")
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM detecciones ORDER BY id DESC LIMIT 1")
        resultado = cur.fetchone()
        cur.close()
        conn.close()
        
        if resultado:
            # Convertir datetime a string
            resultado['fecha_hora'] = resultado['fecha_hora'].isoformat()
            return jsonify(resultado)
        else:
            return jsonify({'mensaje': 'No hay registros'}), 404
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    print("GET /health - Health check")
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
