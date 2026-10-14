"""Obtención de puntos clave faciales y corporales con MediaPipe.

Usa la **Tasks API** (``mediapipe.tasks``), que es la única disponible en
MediaPipe 1.x: el antiguo ``mp.solutions.face_mesh`` ya no existe.

Índices
-------
Los índices del Face Mesh siguen la convención de MediaPipe, que describe la
cara *como si se viera en un espejo*. En una imagen de cámara sin espejar, el
lado izquierdo de la imagen es el lado **derecho** del sujeto. Todos los
cálculos de este proyecto trabajan en espacio de pantalla (ya espejado) y
ordenan los pares por su coordenada ``x``, de modo que esta convención no
afecta al signo del ángulo: sólo importa para dibujar cada contorno en su sitio.

Todos los índices y anillos de contorno se verificaron empíricamente contra
una imagen real de rostro durante el desarrollo.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .config import DetectionConfig
from .errors import ModelError

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Índices del Face Mesh (478 puntos: 468 faciales + 10 del iris)
# ---------------------------------------------------------------------------

FACE_LEFT_IRIS = 473          # centro del iris izquierdo del sujeto
FACE_RIGHT_IRIS = 468         # centro del iris derecho del sujeto
FACE_LEFT_IRIS_RING = (473, 474, 475, 476, 477)
FACE_RIGHT_IRIS_RING = (468, 469, 470, 471, 472)

FACE_LEFT_EYE_OUTER = 263     # esquina exterior del ojo izquierdo
FACE_RIGHT_EYE_OUTER = 33     # esquina exterior del ojo derecho
FACE_LEFT_CHEEK = 454         # lado de la cara / oreja izquierda
FACE_RIGHT_CHEEK = 234        # lado de la cara / oreja derecha
FACE_LEFT_BROW_INNER = 300    # cejas (puntos interiores, para la línea)
FACE_RIGHT_BROW_INNER = 105

FACE_NOSE_TIP = 1
FACE_CHIN = 152
FACE_FOREHEAD = 10
FACE_NOSE_BRIDGE = 168

# Pares simétricos que definen el ángulo de inclinación (roll).
ANGLE_SOURCE_PAIRS: dict[str, tuple[int, int]] = {
    "iris": (FACE_RIGHT_IRIS, FACE_LEFT_IRIS),
    "eyes": (FACE_RIGHT_EYE_OUTER, FACE_LEFT_EYE_OUTER),
    "ears": (FACE_RIGHT_CHEEK, FACE_LEFT_CHEEK),
    "brows": (FACE_RIGHT_BROW_INNER, FACE_LEFT_BROW_INNER),
}

# Pares alternativos si el modelo no incluye los 10 puntos del iris.
IRIS_FALLBACK_PAIR = (FACE_RIGHT_EYE_OUTER, FACE_LEFT_EYE_OUTER)

# Contornos para el dibujado (anillos cerrados).
FACE_CONTOURS: dict[str, tuple[int, ...]] = {
    "left_eye": (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466),
    "right_eye": (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
    "left_brow": (276, 283, 282, 295, 285, 300, 293, 334, 296, 336),
    "right_brow": (46, 53, 52, 65, 55, 70, 63, 105, 66, 107),
    "lips_outer": (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185),
    "lips_inner": (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191),
    "face_oval": (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
                  400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109),
}

# Índices del Pose (33 puntos).
POSE_NOSE = 0
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_SHOULDER_PAIR = (POSE_RIGHT_SHOULDER, POSE_LEFT_SHOULDER)

# Esqueleto reducido: se dibujan sólo tronco y brazos, que son los relevantes
# para la postura. El esqueleto completo (piernas, etc.) satura la pantalla.
POSE_BONES: tuple[tuple[int, int], ...] = (
    (11, 12), (0, 11), (0, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24),
)


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FacePoint:
    """Punto facial normalizado (0-1) convertido a píxeles de pantalla."""

    x: float
    y: float
    z: float = 0.0

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class PosePoint:
    """Punto corporal normalizado (0-1) convertido a píxeles de pantalla."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class LandmarkSet:
    """Resultado de la detección en un fotograma.

    Los puntos ya están en **píxeles de espacio de pantalla**.
    """

    face: tuple[FacePoint, ...] | None
    pose: tuple[PosePoint, ...] | None
    frame_width: int
    frame_height: int

    @property
    def has_face(self) -> bool:
        return bool(self.face)

    @property
    def has_pose(self) -> bool:
        return bool(self.pose)

    @property
    def has_iris(self) -> bool:
        """True si el modelo incluye los 10 puntos del iris (478 en total)."""
        return bool(self.face) and len(self.face) > FACE_LEFT_IRIS  # type: ignore[arg-type]


class LandmarkSource(Protocol):
    """Interfaz que consume :class:`~posture.metrics.PostureEstimator`.

    Existe como :class:`typing.Protocol` para que las pruebas puedan inyectar
    landmarks sintéticos sin necesidad de cámara ni de modelos.
    """

    def detect(self, frame) -> LandmarkSet:
        """Detecta puntos clave en un fotograma BGR."""
        ...

    def close(self) -> None:
        """Libera los recursos del detector."""
        ...


# ---------------------------------------------------------------------------
# Descarga de modelos
# ---------------------------------------------------------------------------


def ensure_model(path: Path, url: str | None, auto_download: bool = True, min_bytes: int = 100_000) -> Path:
    """Se asegura de que el modelo existe en ``path``; si no, lo descarga.

    La descarga se hace a un fichero temporal y se renombra al terminar, de
    modo que una descarga interrumpida nunca deje un modelo corrupto que la
    aplicación intente usar después.
    """
    if path.is_file() and path.stat().st_size >= min_bytes:
        return path

    if not auto_download:
        raise ModelError(
            f"Falta el modelo '{path.name}' y la descarga automática está desactivada.\n"
            f"Descárgalo manualmente de:\n  {url}\n"
            f"y guárdalo en: {path}"
        )
    if not url:
        raise ModelError(f"Falta el modelo '{path.name}' y no hay URL de descarga configurada.")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    log.info("Descargando modelo %s desde %s", path.name, url)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            last_report = 0.0
            while chunk := response.read(1 << 16):
                out.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if now - last_report > 1.0:
                    pct = f"{100.0 * done / total:5.1f}%" if total else f"{done >> 10} KiB"
                    log.info("  %s / %s", pct, path.name)
                    last_report = now
        if tmp.stat().st_size < min_bytes:
            raise ModelError(f"La descarga de '{path.name}' parece incompleta ({tmp.stat().st_size} bytes).")
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise ModelError(
            f"No se pudo descargar '{path.name}' desde {url}\n"
            f" Motivo: {exc}\n"
            "Comprueba la conexión, o descarga el fichero a mano y colócalo en la carpeta de modelos."
        ) from exc
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise ModelError(f"Error de E/S al guardar '{path}': {exc}") from exc

    tmp.replace(path)
    log.info("Modelo listo: %s (%.1f MiB)", path, path.stat().st_size / (1 << 20))
    return path


# ---------------------------------------------------------------------------
# Implementación con MediaPipe
# ---------------------------------------------------------------------------


def _silence_mediapipe_logging() -> None:
    """Reduce el ruido de C++/absl que escribe en la consola.

    Debe llamarse **antes** de importar mediapipe.
    """
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ABSL_LOG_LEVEL", "3")


class MediaPipeLandmarkSource:
    """Detector de Face Landmarker + Pose Landmarker.

    Notas de implementación
    -----------------------
    * La Tasks API exige **marcas de tiempo estrictamente crecientes** en
      ``detect_for_video``; se garantiza aquí con un contador monotónico.
    * El import de ``mediapipe`` es perezoso para que el resto de módulos (y
      las pruebas) puedan usarse sin la librería nativa.
    * Las tareas nativas no son thread-safe: el objeto se protege con un lock.
    """

    def __init__(self, cfg: DetectionConfig, base_dir: Path | str = "."):
        self._cfg = cfg
        self._lock = threading.Lock()
        self._last_timestamp_ms = -1
        self._closed = False
        self._face = None
        self._pose = None

        _silence_mediapipe_logging()
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise ModelError(
                "No se pudo importar 'mediapipe'.\n"
                "Instala las dependencias con:  pip install -r requirements.txt"
            ) from exc

        models_dir = Path(base_dir) / cfg.models_dir
        face_path = ensure_model(
            models_dir / cfg.face_model,
            cfg.model_urls.get(cfg.face_model),
            cfg.auto_download,
        )
        delegate = self._delegate(BaseOptions, cfg.delegate)

        try:
            self._face = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(face_path), delegate=delegate),
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=1,
                    min_face_detection_confidence=cfg.min_face_detection_confidence,
                    min_face_presence_confidence=cfg.min_face_detection_confidence,
                    min_tracking_confidence=cfg.min_face_detection_confidence,
                )
            )
        except Exception as exc:  # pragma: no cover
            raise ModelError(f"No se pudo cargar el modelo facial '{face_path.name}': {exc}") from exc

        if cfg.use_pose:
            pose_path = ensure_model(
                models_dir / cfg.pose_model,
                cfg.model_urls.get(cfg.pose_model),
                cfg.auto_download,
            )
            try:
                self._pose = vision.PoseLandmarker.create_from_options(
                    vision.PoseLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=str(pose_path), delegate=delegate),
                        running_mode=vision.RunningMode.VIDEO,
                        num_poses=1,
                        min_pose_detection_confidence=cfg.min_pose_detection_confidence,
                        min_presence_confidence=cfg.min_pose_detection_confidence,
                        min_tracking_confidence=cfg.min_pose_detection_confidence,
                    )
                )
            except Exception as exc:  # pragma: no cover
                # El trunk es una mejora, no un requisito: se degrada sin él.
                log.warning("No se pudo cargar el modelo de pose, se seguirá sin referencia de hombros: %s", exc)
                self._pose = None

        self._mp = mp
        self._vision = vision
        log.info("Modelos cargados (delegate=%s, pose=%s)", cfg.delegate, self._pose is not None)

    # -- interno ------------------------------------------------------------
    @staticmethod
    def _delegate(base_options_cls, requested: str):
        """Traduce 'cpu'/'gpu' al enumerado de la Tasks API."""
        return base_options_cls.Delegate.GPU if requested == "gpu" else base_options_cls.Delegate.CPU

    def _next_timestamp_ms(self) -> int:
        ts = time.monotonic_ns() // 1_000_000
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1  # la API exige monotonicidad estricta
        self._last_timestamp_ms = ts
        return ts

    # -- API ----------------------------------------------------------------
    def detect(self, frame) -> LandmarkSet:
        """Detecta rostro (y opcionalmente tronco) en un fotograma BGR."""
        import cv2

        if self._closed:
            raise ModelError("El detector se ha cerrado")
        if self._face is None:  # pragma: no cover
            raise ModelError("El detector facial no está inicializado")

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp = self._next_timestamp_ms()

        with self._lock:
            face_result = self._face.detect_for_video(image, timestamp)
            pose_result = self._pose.detect_for_video(image, timestamp) if self._pose else None

        # Escala normalizada (0-1) -> píxeles. Se usa (w-1)/(h-1) para que el
        # punto 1.0 caiga dentro del último píxel y no se salga del fotograma.
        sx = (width - 1) if width > 1 else 1.0
        sy = (height - 1) if height > 1 else 1.0

        face_pts: tuple[FacePoint, ...] | None = None
        if face_result.face_landmarks:
            face_pts = tuple(
                FacePoint(p.x * sx, p.y * sy, getattr(p, "z", 0.0)) for p in face_result.face_landmarks[0]
            )

        pose_pts: tuple[PosePoint, ...] | None = None
        if pose_result is not None and pose_result.pose_landmarks:
            pose_pts = tuple(
                PosePoint(p.x * sx, p.y * sy, getattr(p, "z", 0.0), float(getattr(p, "visibility", 1.0)))
                for p in pose_result.pose_landmarks[0]
            )

        return LandmarkSet(face=face_pts, pose=pose_pts, frame_width=width, frame_height=height)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            for task in (self._face, self._pose):
                if task is not None:
                    try:
                        task.close()
                    except Exception:  # pragma: no cover - cierre best-effort
                        log.debug("Error al cerrar un detector", exc_info=True)
            self._face = None
            self._pose = None

    def __enter__(self) -> "MediaPipeLandmarkSource":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def build_landmark_source(cfg: DetectionConfig, base_dir: Path | str = ".") -> LandmarkSource:
    """Fábrica del detector real (punto único de importación de MediaPipe)."""
    return MediaPipeLandmarkSource(cfg, base_dir)


def face_pair_indices(angle_source: str, has_iris: bool) -> tuple[int, int]:
    """Índices del par simétrico a usar según la fuente de ángulo elegida.

    Si se pide ``iris`` pero el modelo no trae los puntos del iris, se cae a las
    esquinas de los ojos en lugar de fallar.
    """
    if angle_source == "iris" and not has_iris:
        log.warning("El modelo no incluye puntos de iris; se usan las esquinas de los ojos")
        return IRIS_FALLBACK_PAIR
    return ANGLE_SOURCE_PAIRS.get(angle_source, ANGLE_SOURCE_PAIRS["eyes"])


def contour_points(contour: Sequence[int], face: Sequence[FacePoint]) -> list[tuple[int, int]]:
    """Convierte un contorno del Face Mesh en puntos de píxel dibujables."""
    return [(int(round(face[i].x)), int(round(face[i].y))) for i in contour if i < len(face)]
