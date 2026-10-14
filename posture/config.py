"""Configuración declarativa de la aplicación.

Toda la parametrización vive en dataclasses anidadas que se pueden:

* cargar de ``config.json`` (las claves ausentes usan el valor por defecto),
* sobrescribir desde la línea de órdenes con ``--set seccion.clave=valor``,
* validar con rangos coherentes, y
* volver a guardar con ``--save-config``.

Se usa ``config.json`` plano (sin comentarios) porque el estándar ``json`` de
Python no los admite; la documentación de cada parámetro vive aquí y en el
README.
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union, get_args, get_origin

from .errors import ConfigError

# ---------------------------------------------------------------------------
# Secciones
# ---------------------------------------------------------------------------


@dataclass
class CameraConfig:
    """Captura de vídeo.

    Attributes
    ----------
    source:
        ``"0"``, ``"1"``... para una webcam (índice del dispositivo), o una
        ruta a un vídeo/imagen para reproducirla. Permite validar el monitor
        sin cámara (por ejemplo, con una grabación).
    width, height, fps:
        Resolución y cadencia solicitadas. Las webcams suelen ignorarlas y
        entregan su propia resolución; la app se adapta al tamaño real.
    mirror:
        Modo espejo (imagen espejada), que es lo natural para monitorizarse
        uno a sí mismo.
    backend:
        Backend de captura de OpenCV: ``auto``, ``dshow`` (Windows),
        ``msmf`` (Windows) o ``any``.
    fallback_indices:
        Índices a probar si el principal falla. Muy útil en Windows, donde el
        índice 0 suele estar ocupado por otra aplicación.
    """

    source: str = "0"
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True
    backend: str = "auto"
    fallback_indices: list[int] = field(default_factory=lambda: [1, 2])


@dataclass
class DetectionConfig:
    """Detección de puntos clave con MediaPipe (Tasks API).

    Attributes
    ----------
    models_dir:
        Carpeta donde se guardan/leen los modelos ``.task``.
    angle_source:
        Qué línea facial define la inclinación (roll):

        * ``iris``   -> línea entre los centros de ambos iris. **Recomendado**:
          los iris son los puntos más estables del Face Mesh.
        * ``eyes``   -> línea entre las esquinas exteriores de los ojos.
        * ``ears``   -> línea entre los dos lados de la cara (orejas/mejillas).
        * ``brows``  -> línea entre las cejas.
    reference:
        Referencia contra la que se mide la inclinación:

        * ``auto``       -> usa la línea de hombros si es visible y fiable, y
          cae a la línea facial si el tronco no se ve. **Recomendado**.
        * ``shoulders``  -> exige la línea de hombros (medición de flexión
          lateral cervical respecto al tronco).
        * ``face_only``  -> mide en el marco de la cámara, sin compensar la
          inclinación de la cámara o del tronco.
    inference_width:
        Ancho al que se reduce el fotograma antes de la inferencia. 640 px es
        el punto óptimo: por debajo se pierde precisión en los landmarks y por
        encima solo se pierde CPU (los modelos son muy rápidos).
    min_face_span_ratio:
        Longitud mínima de la línea facial, como fracción del ancho del
        fotograma. Filtra detecciones diminutas o degeneradas.
    edge_margin_px:
        Margen respecto al borde del fotograma. Una cara parcialmente fuera de
        cuadro produce ángulos poco fiables, así que se descarta.
    """

    models_dir: str = "models"
    face_model: str = "face_landmarker.task"
    pose_model: str = "pose_landmarker_lite.task"
    auto_download: bool = True
    use_pose: bool = True
    delegate: str = "cpu"
    angle_source: str = "iris"
    reference: str = "auto"
    inference_width: int = 640
    min_face_detection_confidence: float = 0.5
    min_pose_detection_confidence: float = 0.5
    shoulder_visibility: float = 0.5
    min_face_span_ratio: float = 0.04
    edge_margin_px: int = 6
    reference_hysteresis_frames: int = 15

    model_urls: dict[str, str] = field(
        default_factory=lambda: {
            "face_landmarker.task": (
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task"
            ),
            "pose_landmarker_lite.task": (
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            ),
        }
    )


@dataclass
class SmoothingConfig:
    """Suavizado temporal (filtro One Euro).

    ``min_cutoff`` bajo = más suavizado y menos ruido, pero más retardo.
    ``beta`` alto = sigue mejor los movimientos rápidos. Si los usuarios se
    quejan de que la alerta «late», subir ``beta``; si el indicador parpadea,
    bajar ``min_cutoff``.
    """

    enabled: bool = True
    min_cutoff: float = 1.1
    beta: float = 0.35
    d_cutoff: float = 1.0
    shoulder_min_cutoff: float = 0.5
    shoulder_beta: float = 0.15


@dataclass
class ThresholdsConfig:
    """Umbrales de alerta.

    Attributes
    ----------
    tilt_degrees:
        Desviación tolerada respecto al ángulo neutro. Valores 5-7 son
        razonables; por debajo de 4 puede ser demasiado sensible.
    tolerance_seconds:
        Tiempo **continuo** fuera de rango antes de avisar. Es lo que
        distingue un aviso útil de uno molesto.
    hysteresis_degrees:
        Banda muerta: el aviso sólo se limpia al bajar de
        ``tilt_degrees - hysteresis_degrees``. Evita el parpadeo cuando la
        cabeza ronda el umbral.
    reset_dwell_on_face_loss:
        Si se pierde el rostro, reiniciar el cronómetro. Es lo prudente: no
        queremos que al volver a aparecer la cara salte una alerta acumulada.
    forward_lean_enabled:
        Comprobación secundaria de inclinación frontal (cabeza por delante de
        los hombros). Desactivada por defecto porque es una aproximación
        mucho más ruidosa que el roll lateral.
    """

    tilt_degrees: float = 7.0
    tolerance_seconds: float = 3.0
    hysteresis_degrees: float = 1.5
    reset_dwell_on_face_loss: bool = True
    forward_lean_enabled: bool = False
    forward_lean_degrees: float = 15.0
    forward_lean_tolerance_seconds: float = 6.0


@dataclass
class CalibrationConfig:
    """Calibración del ángulo neutro.

    Se toma la **mediana** de los ángulos de los últimos ``sample_seconds``,
    lo que hace el resultado robusto a los fotogramas ruidosos.
    """

    auto_on_start: bool = True
    countdown_seconds: float = 3.0
    sample_seconds: float = 1.5
    min_samples: int = 10
    manual_zero_degrees: float | None = None


@dataclass
class FeedbackConfig:
    """Interfaz y avisos.

    ``sound_kind``:

    * ``system`` -> sonido de sistema del sistema operativo (suave, no
      estridente). Recomendado.
    * ``tone``   -> tono senoidal de la frecuencia indicada.
    * ``bell``   -> carácter de campana de terminal.
    """

    window_name: str = "Monitor de postura de cabeza"
    window_scale: float = 0.7
    fullscreen: bool = False
    show_face_mesh: bool = True
    show_pose: bool = True
    show_hud: bool = True
    show_timer_bar: bool = True
    show_gauge: bool = True
    sound_enabled: bool = True
    sound_kind: str = "system"
    sound_repeat_seconds: float = 8.0
    tone_frequency_hz: int = 880
    tone_duration_ms: int = 220


@dataclass
class KeysConfig:
    """Atajos de teclado (un solo carácter, sin distinguir mayúsculas)."""

    calibrate: str = "c"
    quit: str = "q"
    increase_sensitivity: str = "+"
    decrease_sensitivity: str = "-"
    reset_sensitivity: str = "0"
    toggle_sound: str = "m"
    toggle_mesh: str = "s"
    toggle_hud: str = "h"
    pause: str = "p"
    fullscreen: str = "f"
    sensitivity_step_degrees: float = 1.0


@dataclass
class AppConfig:
    """Configuración raíz."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    keys: KeysConfig = field(default_factory=KeysConfig)

    # Modo de ejecución
    headless: bool = False           # sin ventana: sólo salida por consola
    duration_seconds: float = 0.0    # 0 = hasta pulsar 'q' (parar a los N s)
    log_level: str = "info"


# ---------------------------------------------------------------------------
# Coerción genérica de tipos
# ---------------------------------------------------------------------------

_TRUE = {"1", "true", "yes", "y", "on", "si", "sí"}
_FALSE = {"0", "false", "no", "n", "off"}


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    """Convierte ``value`` (venido de JSON o de la CLI) al tipo ``annotation``."""
    # Optional[X] / Union[...]
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if value is None:
            return None
        errors = []
        for arg in args:
            try:
                return _coerce(value, arg, path)
            except (ValueError, TypeError, ConfigError) as exc:  # pragma: no cover
                errors.append(str(exc))
        raise ConfigError(f"'{path}': valor {value!r} no válido (se esperaba {args})")

    if dataclasses.is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ConfigError(f"'{path}': se esperaba una tabla de opciones, no {type(value).__name__}")
        return _build(annotation, value, path)

    if origin in (list, tuple) or annotation in (list, tuple):
        item_type = str
        if origin in (list, tuple):
            args = get_args(annotation)
            item_type = args[0] if args else str
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"'{path}': se esperaba una lista")
        return [_coerce(v, item_type, f"{path}[{i}]") for i, v in enumerate(value)]

    if origin is dict or annotation is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"'{path}': se esperaba una tabla de opciones")
        args = get_args(annotation) if origin is dict else ()
        key_t = args[0] if args else str
        val_t = args[1] if len(args) > 1 else Any
        return {_coerce(k, key_t, path): _coerce(v, val_t, f"{path}.{k}") for k, v in value.items()}

    if annotation is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            low = value.strip().lower()
            if low in _TRUE:
                return True
            if low in _FALSE:
                return False
        raise ConfigError(f"'{path}': '{value}' no es un booleano (usa true/false)")

    if annotation is int:
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ConfigError(f"'{path}': '{value}' no es un entero") from None

    if annotation is float:
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"'{path}': '{value}' no es un número") from None

    if annotation is str:
        return str(value)

    return value  # anotaciones no primitivas: se pasan tal cual


def _build(cls: type, data: dict, prefix: str = "") -> Any:
    """Construye la dataclass ``cls`` a partir del diccionario ``data``."""
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = [k for k in data if k not in fields]
    if unknown:
        known = ", ".join(sorted(fields))
        raise ConfigError(
            f"opción desconocida: {', '.join(prefix + u for u in unknown)}. "
            f"Opciones válidas en '{prefix or 'raíz'}': {known}"
        )
    kwargs: dict[str, Any] = {}
    for name, f in fields.items():
        if name not in data:
            continue
        kwargs[name] = _coerce(data[name], f.type if not isinstance(f.type, str) else _resolve(f.type, cls),
                               f"{prefix}{name}")
    return cls(**kwargs)

_ANNOTATION_NS = {c.__name__: c for c in (
    CameraConfig, DetectionConfig, SmoothingConfig, ThresholdsConfig,
    CalibrationConfig, FeedbackConfig, KeysConfig,
)}


def _resolve(annotation: str, owner: type) -> Any:
    """Resuelve anotaciones escritas como cadena (por ``from __future__``)."""
    namespace = {**vars(typing), **_ANNOTATION_NS, "Any": Any}
    try:
        return eval(annotation, namespace)  # noqa: S307 - namespace cerrada y propia
    except Exception:
        pass
    if annotation in _ANNOTATION_NS:
        return _ANNOTATION_NS[annotation]
    if annotation.startswith(("list[", "List[")):
        return list
    if annotation.startswith(("dict[", "Dict[")):
        return dict
    if annotation.startswith(("str | None", "Optional[str]")):
        return str | None
    return Any


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------

_RANGES: list[tuple[str, float, float]] = [
    ("camera.width", 160, 7680),
    ("camera.height", 120, 4320),
    ("camera.fps", 1, 240),
    ("detection.inference_width", 160, 3840),
    ("detection.min_face_detection_confidence", 0.0, 1.0),
    ("detection.min_pose_detection_confidence", 0.0, 1.0),
    ("detection.shoulder_visibility", 0.0, 1.0),
    ("detection.min_face_span_ratio", 0.0, 0.9),
    ("detection.edge_margin_px", 0, 200),
    ("smoothing.min_cutoff", 0.05, 50.0),
    ("smoothing.beta", 0.0, 10.0),
    ("thresholds.tilt_degrees", 0.5, 60.0),
    ("thresholds.tolerance_seconds", 0.1, 300.0),
    ("thresholds.hysteresis_degrees", 0.0, 30.0),
    ("thresholds.forward_lean_degrees", 1.0, 60.0),
    ("thresholds.forward_lean_tolerance_seconds", 0.1, 300.0),
    ("calibration.countdown_seconds", 0.0, 30.0),
    ("calibration.sample_seconds", 0.1, 30.0),
    ("calibration.min_samples", 1, 2000),
    ("feedback.window_scale", 0.2, 4.0),
    ("feedback.sound_repeat_seconds", 0.5, 600.0),
    ("feedback.tone_frequency_hz", 100, 8000),
    ("feedback.tone_duration_ms", 20, 5000),
    ("keys.sensitivity_step_degrees", 0.1, 20.0),
    ("duration_seconds", 0.0, 86400.0),
]

_ENUMS: list[tuple[str, set[str]]] = [
    ("camera.backend", {"auto", "dshow", "msmf", "any"}),
    ("detection.delegate", {"cpu", "gpu"}),
    ("detection.angle_source", {"iris", "eyes", "ears", "brows"}),
    ("detection.reference", {"auto", "shoulders", "face_only"}),
    ("feedback.sound_kind", {"system", "tone", "bell"}),
    ("log_level", {"debug", "info", "warning", "error"}),
]


def _get_path(cfg: AppConfig, dotted: str) -> Any:
    obj: Any = cfg
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _set_path(cfg: AppConfig, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    obj: Any = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def validate(cfg: AppConfig) -> AppConfig:
    """Comprueba rangos y valores enumerados. Devuelve la config o lanza
    :class:`~posture.errors.ConfigError` con un mensaje accionable."""
    for dotted, lo, hi in _RANGES:
        value = _get_path(cfg, dotted)
        if not (lo <= value <= hi):
            raise ConfigError(f"'{dotted}' = {value} está fuera del rango admitido [{lo}, {hi}]")

    for dotted, allowed in _ENUMS:
        value = _get_path(cfg, dotted)
        if value not in allowed:
            raise ConfigError(f"'{dotted}' = '{value}' no válido; opciones: {', '.join(sorted(allowed))}")

    if cfg.thresholds.hysteresis_degrees >= cfg.thresholds.tilt_degrees:
        raise ConfigError(
            "thresholds.hysteresis_degrees debe ser menor que thresholds.tilt_degrees "
            f"({cfg.thresholds.hysteresis_degrees} >= {cfg.thresholds.tilt_degrees}); "
            "si no, la banda muerta cubriría todo el rango y nunca se limpiaría la alerta"
        )

    if cfg.detection.reference in ("shoulders", "auto") and not cfg.detection.use_pose:
        raise ConfigError(
            "detection.reference = 'auto' o 'shoulders' requiere detection.use_pose = true"
        )

    if cfg.detection.angle_source == "iris" and not cfg.detection.face_model:
        raise ConfigError("detection.face_model no puede estar vacío")

    for key in dataclasses.fields(KeysConfig):
        value = getattr(cfg.keys, key)
        if key.endswith(("_seconds", "_degrees", "step_degrees")):
            continue
        if not isinstance(value, str) or len(value) != 1:
            raise ConfigError(f"keys.{key} debe ser un único carácter, no '{value}'")

    return cfg


# ---------------------------------------------------------------------------
# Carga / guardado / overrides
# ---------------------------------------------------------------------------


def default_config() -> AppConfig:
    """Configuración por defecto (sin leer ficheros)."""
    return AppConfig()


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> AppConfig:
    """Carga la configuración aplicando, en este orden:

    1. valores por defecto,
    2. el contenido de ``path`` (si se indica),
    3. los overrides ``--set seccion.clave=valor`` de la línea de órdenes.

    Parámetros
    ----------
    path:
        Ruta a ``config.json``. Si es ``None`` sólo se usan los valores por
        defecto y los overrides.
    overrides:
        Lista de cadenas ``"umbral.tilt_degrees=5"``.
    """
    cfg = default_config()

    if path is not None:
        cfg_path = Path(path)
        if not cfg_path.is_file():
            raise ConfigError(
                f"No se encontró el archivo de configuración '{cfg_path}'. "
                "Créalo con --print-config > config.json, o ejecutes sin --config."
            )
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"'{cfg_path}' no es un JSON válido: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"'{cfg_path}' debe contener una tabla de opciones en la raíz")
        cfg = _build(AppConfig, raw)

    if overrides:
        apply_overrides(cfg, overrides)

    return validate(cfg)


def apply_overrides(cfg: AppConfig, overrides: list[str]) -> AppConfig:
    """Aplica overrides ``seccion.clave=valor`` sobre ``cfg`` (in-place)."""
    fields_cache = {
        "camera": CameraConfig, "detection": DetectionConfig, "smoothing": SmoothingConfig,
        "thresholds": ThresholdsConfig, "calibration": CalibrationConfig,
        "feedback": FeedbackConfig, "keys": KeysConfig,
    }
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override '{item}' con formato incorrecto; se espera 'seccion.clave=valor'")
        dotted, raw_value = item.split("=", 1)
        dotted, raw_value = dotted.strip(), raw_value.strip()
        parts = dotted.split(".")
        section = parts[0] if len(parts) > 1 else "__root__"
        key = parts[-1]
        if section == "__root__":
            owner, field_name = AppConfig, key
            valid = {f.name for f in dataclasses.fields(AppConfig)}
            if key not in valid:
                raise ConfigError(
                    f"override '{item}': sección desconocida. Secciones: "
                    f"{', '.join(sorted(fields_cache))}, o raíz: {', '.join(sorted(valid))}"
                )
        else:
            if section not in fields_cache:
                raise ConfigError(
                    f"override '{item}': sección desconocida '{section}'. "
                    f"Secciones: {', '.join(sorted(fields_cache))}"
                )
            owner = fields_cache[section]
            field_name = key
            valid = {f.name for f in dataclasses.fields(owner)}
            if field_name not in valid:
                raise ConfigError(f"override '{item}': '{section}' no tiene la opción '{field_name}'")
        try:
            ftype = {f.name: f.type for f in dataclasses.fields(owner)}[field_name]
            if isinstance(ftype, str):
                ftype = _resolve(ftype, owner)
            value = _coerce(raw_value, ftype, dotted)
        except ConfigError as exc:
            raise ConfigError(f"override '{item}': {exc}") from None
        _set_path(cfg, dotted, value)

    return validate(cfg)


def to_dict(cfg: AppConfig) -> dict[str, Any]:
    """Convierte la configuración a un diccionario plano de JSON."""
    return dataclasses.asdict(cfg)


def save_config(cfg: AppConfig, path: str | Path) -> Path:
    """Guarda la configuración como JSON indentado."""
    out = Path(path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_dict(cfg), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def format_config(cfg: AppConfig) -> str:
    """Configuración como JSON listo para copiar en un ``config.json``."""
    return json.dumps(to_dict(cfg), indent=2, ensure_ascii=False)
