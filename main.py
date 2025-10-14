import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from werkzeug.utils import secure_filename
from collections import Counter
import io
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Cargar modelo YOLO
print("Cargando YOLO...")
model = YOLO('yolov8n.pt')
print("YOLO listo!")

# REEMPLAZA CON TU CONTRASEÑA DE SUPABASE
DATABASE_URL = "postgresql://postgres:TU_CONTRASEÑA_AQUI@db.xkmbefvvasnnqjapvjdt.supabase.co:5432/postgres"

def get_db():
    return psycopg2.connect(DATABASE_URL)

def calcular_forma(x, y, w, h, contorno=None):
    ratio = w / h if h > 0 else 1
    if 0.9 <= ratio <= 1.1:
        return "cuadrado"
    if contorno is not None and len(contorno) >= 5:
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        if perimetro > 0:
            circularidad = (4 * np.pi * area) / (perimetro ** 2)
            if circularidad >= 0.85:
                return "circulo"
    return "rectangulo"

def calcular_color(imagen, x, y, w, h):
    roi = imagen[y:y+h, x:x+w]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pixels = roi_rgb.reshape(-1, 3)
    pixels_list = [tuple(p) for p in pixels]
    color_counts = Counter(pixels_list)
    r, g, b = color_counts.most_common(1)[0][0]
    
    colores = {
        'rojo': (255, 0, 0), 'verde': (0, 255, 0), 'azul': (0, 0, 255),
        'amarillo': (255, 255, 0), 'naranja': (255, 165, 0), 'morado': (128, 0, 128),
        'rosa': (255, 192, 203), 'negro': (0, 0, 0), 'blanco': (255, 255, 255),
        'gris': (128, 128, 128), 'cafe': (165, 42, 42), 'celeste': (135, 206, 235)
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
    return '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detector YOLO</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .upload-section {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            border: 2px dashed #667eea;
        }
        .btn {
            background: #667eea;
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s;
            font-weight: 600;
        }
        .btn:hover {
            background: #5568d3;
            transform: translateY(-2px);
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        #preview {
            max-width: 100%;
            max-height: 400px;
            border-radius: 10px;
            display: none;
            margin: 20px auto;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .results {
            background: #f0f7ff;
            padding: 25px;
            border-radius: 15px;
            margin-top: 20px;
            display: none;
            border-left: 5px solid #667eea;
        }
        .result-item {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .result-label {
            font-weight: 600;
            color: #667eea;
        }
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .error {
            background: #fee;
            color: #c33;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            display: none;
        }
        input[type="file"] {
            display: none;
        }
        .file-label {
            display: block;
            width: 100%;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Detector de Objetos YOLO</h1>
        <p class="subtitle">Sube una imagen para detectar objetos, formas y colores</p>
        
        <div class="upload-section">
            <label for="fileInput" class="file-label">
                <button class="btn" type="button" onclick="document.getElementById('fileInput').click()">
                    📁 Seleccionar Imagen
                </button>
            </label>
            <input type="file" id="fileInput" accept="image/*" onchange="previewImage(event)">
            
            <img id="preview" alt="Vista previa">
            
            <button class="btn" id="detectBtn" onclick="detectar()" style="margin-top: 15px; display: none;">
                🚀 Detectar Objeto
            </button>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 15px; color: #667eea; font-weight: 600;">Procesando imagen...</p>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="results" id="results">
            <h3>📊 Resultados</h3>
            <div class="result-item">
                <span class="result-label">Objeto:</span>
                <span id="objeto">-</span>
            </div>
            <div class="result-item">
                <span class="result-label">Forma:</span>
                <span id="forma">-</span>
            </div>
            <div class="result-item">
                <span class="result-label">Color:</span>
                <span id="color">-</span>
            </div>
        </div>

        <div style="margin-top: 30px; text-align: center;">
            <button class="btn" onclick="cargarUltimo()" style="max-width: 300px; margin: 0 auto;">
                🔄 Ver Último Registro
            </button>
            <div id="lastResults" style="margin-top: 15px;"></div>
        </div>
    </div>

    <script>
        let selectedFile = null;

        function previewImage(event) {
            const file = event.target.files[0];
            if (file) {
                selectedFile = file;
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('preview').src = e.target.result;
                    document.getElementById('preview').style.display = 'block';
                    document.getElementById('detectBtn').style.display = 'block';
                };
                reader.readAsDataURL(file);
                document.getElementById('results').style.display = 'none';
                document.getElementById('error').style.display = 'none';
            }
        }

        async function detectar() {
            if (!selectedFile) {
                mostrarError('Selecciona una imagen');
                return;
            }

            const formData = new FormData();
            formData.append('imagen', selectedFile);

            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            document.getElementById('detectBtn').disabled = true;

            try {
                const response = await fetch('/detectar', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    document.getElementById('objeto').textContent = data.objeto;
                    document.getElementById('forma').textContent = data.forma;
                    document.getElementById('color').textContent = data.color;
                    document.getElementById('results').style.display = 'block';
                } else {
                    mostrarError(data.error || 'Error en detección');
                }
            } catch (error) {
                mostrarError('Error: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('detectBtn').disabled = false;
            }
        }

        function mostrarError(mensaje) {
            document.getElementById('error').textContent = '⚠️ ' + mensaje;
            document.getElementById('error').style.display = 'block';
        }

        async function cargarUltimo() {
            try {
                const response = await fetch('/api/ultimo');
                const data = await response.json();

                if (response.ok && data.id) {
                    const html = `
                        <div style="background: #fff9e6; padding: 20px; border-radius: 10px; margin-top: 15px;">
                            <p><strong>Objeto:</strong> ${data.objeto}</p>
                            <p><strong>Forma:</strong> ${data.forma}</p>
                            <p><strong>Color:</strong> ${data.color}</p>
                            <p><strong>Fecha:</strong> ${new Date(data.fecha_hora).toLocaleString()}</p>
                        </div>
                    `;
                    document.getElementById('lastResults').innerHTML = html;
                } else {
                    document.getElementById('lastResults').innerHTML = '<p style="color: #666;">No hay registros</p>';
                }
            } catch (error) {
                document.getElementById('lastResults').innerHTML = '<p style="color: #c33;">Error al cargar</p>';
            }
        }
    </script>
</body>
</html>
    '''

@app.route('/detectar', methods=['POST'])
def detectar():
    print("Detectando...")
    try:
        if 'imagen' not in request.files:
            return jsonify({'error': 'No hay imagen'}), 400
        
        file = request.files['imagen']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        imagen = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        resultados = model(imagen)
        
        if len(resultados[0].boxes) == 0:
            return jsonify({'error': 'No se detectaron objetos'}), 400
        
        box = resultados[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        objeto = model.names[int(box.cls[0])]
        
        w, h = x2 - x1, y2 - y1
        
        roi_gray = cv2.cvtColor(imagen[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(roi_gray, 127, 255, cv2.THRESH_BINARY)
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contorno = contornos[0] if contornos else None
        
        forma = calcular_forma(x1, y1, w, h, contorno)
        color = calcular_color(imagen, x1, y1, w, h)
        
        print(f"Detectado: {objeto}, {forma}, {color}")
        
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO detecciones (objeto, forma, color, fecha_hora, imagen_url) VALUES (%s, %s, %s, %s, %s)",
            (objeto, forma, color, datetime.now(), "memoria")
        )
        conn.commit()
        cur.close()
        conn.close()
        
        print("Guardado en BD")
        
        return jsonify({
            'objeto': objeto,
            'forma': forma,
            'color': color
        })
        
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ultimo')
def ultimo():
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM detecciones ORDER BY id DESC LIMIT 1")
        resultado = cur.fetchone()
        cur.close()
        conn.close()
        
        if resultado:
            resultado['fecha_hora'] = resultado['fecha_hora'].isoformat()
            return jsonify(resultado)
        return jsonify({'mensaje': 'No hay registros'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
