"""
Monitor de postura de cabeza.

Paquete modular del monitor de inclinación cervical basado en MediaPipe + OpenCV.

Arquitectura (cada módulo tiene una única responsabilidad):

    config      -> carga/valida/guarda la configuración (JSON + CLI)
    geometry    -> matemática de ángulos y filtros de suavizado (One Euro)
    landmarks   -> obtaining de puntos clave (MediaPipe Tasks API) + índices
    metrics     -> landmarks -> métricas postulares (roll e inclinación frontal)
    calibration -> captura del ángulo neutro (ventana + mediana)
    monitor     -> máquina de estados con temporizador de tolerancia
    alerts      -> aviso sonoro (Windows/macOS/Linux)
    overlay     -> dibujado del HUD, esqueleto y banners
    camera      -> captura de vídeo (webcam / archivo / imagen)
    app         -> bucle principal y atajos de teclado
"""

__all__ = [
    "alerts",
    "app",
    "calibration",
    "camera",
    "config",
    "errors",
    "geometry",
    "landmarks",
    "metrics",
    "monitor",
    "overlay",
]

__version__ = "1.0.0"
