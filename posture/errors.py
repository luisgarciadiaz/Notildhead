"""Excepciones propias del monitor, para que el script principal pueda
distinguir claramente un error de configuración de uno de cámara o de modelo.
"""


class PostureError(Exception):
    """Error base de la aplicación."""


class ConfigError(PostureError):
    """El archivo de configuración no existe, tiene tipos erróneos o valores
    fuera de rango admitidos."""


class ModelError(PostureError):
    """Falta el modelo de MediaPipe y no se pudo descargar."""


class VideoSourceError(PostureError):
    """No se pudo abrir la cámara (o el archivo de vídeo) solicitado."""
