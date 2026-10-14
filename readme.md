Actúa como un desarrollador experto en Python, visión por computadora y ergonomía. 
Necesito que crees una aplicación de escritorio modular en Python que utilice la cámara web para monitorear la postura de mi cabeza en tiempo real y alertarme si se inclina o se ladea hacia un lado de forma prolongada.

Requisitos técnicos y funcionales:
1. **Librerías principales:** Utiliza `OpenCV` para la captura de video y `MediaPipe` (Face Mesh o Pose) para la detección de puntos clave faciales y de los hombros.
2. **Cálculo de inclinación (Roll):** 
   - Calcula el ángulo de inclinación de la cabeza tomando como referencia la línea horizontal entre ambos ojos o entre las orejas, o comparándola con la línea horizontal de los hombros.
   - Define un umbral de tolerancia configurable (por ejemplo, una desviación mayor a 5 o 7 grados respecto al eje vertical central durante más de 3 segundos).
3. **Interfaz y Feedback:**
   - Muestra una ventana de video ligera (modo espejo) con el esqueleto o los puntos clave superpuestos.
   - Si la cabeza está recta, muestra el indicador en color verde.
   - Si la cabeza permanece inclinada superando el umbral de tiempo, cambia el indicador a rojo y emite una alerta visual clara (y opcionalmente un sonido suave o tono del sistema).
4. **Controles y Configuración:**
   - Añade un botón o tecla rápida (por ejemplo, la tecla 'C') para realizar una "calibración inicial" que tome la posición neutral actual de mi cabeza como el ángulo cero de referencia.
   - Incluye un archivo de configuración simple (`config.json` o variables al inicio del script) para ajustar la sensibilidad, el tiempo de tolerancia en segundos antes de la alerta, y el índice de la cámara web.
5. **Estructura del proyecto:** 
   - Proporciona el código limpio, comentado y estructurado en un script principal (ej. `head_posture_monitor.py`) junto con un archivo `requirements.txt` con las dependencias exactas (`opencv-python`, `mediapipe`, `numpy`).