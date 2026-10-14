"""Matemática de ángulos y suavizado temporal.

Convenciones de coordenadas
---------------------------
Todo el proyecto trabaja en **espacio de pantalla**: el fotograma ya viene
espejado (``mirror``) y los puntos clave están expresados en píxeles con el
origen en la esquina superior izquierda, tal y como los usa OpenCV.

* El eje ``y`` crece **hacia abajo** (convención de imagen).
* ``x`` crece hacia la derecha.

Con estas convenciones, si ``izq`` es el punto más a la izquierda y ``der`` el
más a la derecha de un par simétrico (ojos, orejas, hombros), entonces::

    angulo = degrees(atan2(y_der - y_izq, x_der - x_izq))

devuelve un valor en ``(-90, 90]`` y su signo tiene un significado físico
directo, tal y como se verificó empíricamente durante el desarrollo:

* ``angulo > 0``  -> el punto de la derecha está más abajo -> **la cabeza se
  ladea hacia la derecha de la pantalla** (que, en modo espejo, es la derecha
  del propio usuario).
* ``angulo < 0``  -> la cabeza se ladea hacia la izquierda.

Por eso ``line_angle_deg`` exige siempre que el primer punto esté a la izquierda
del segundo (invariante a la permuta), en lugar de normalizar el ángulo módulo
180, que perdería el signo.
"""

from __future__ import annotations

import math
from typing import Sequence

# --- Ángulos -----------------------------------------------------------------

Point2 = tuple[float, float]


def degrees(radians: float) -> float:
    """Radianes -> grados."""
    return math.degrees(radians)


def wrap_degrees(angle: float) -> float:
    """Normaliza a ``(-180, 180]``."""
    return (angle + 180.0) % 360.0 - 180.0


def wrap_line_angle(angle: float) -> float:
    """Normaliza el ángulo de una *recta* a ``(-90, 90]``.

    Una recta no tiene orientación: los ángulos que difieren en 180 grados
    describen la misma recta. Es la normalización que hay que aplicar a los
    ángulos de las líneas ocular, auricular y de hombros.
    """
    return (angle + 90.0) % 180.0 - 90.0


def line_angle_deg(left_pt: Point2, right_pt: Point2) -> float:
    """Ángulo de la recta que une dos puntos, en grados y en ``(-90, 90]``.

    Parámetros
    ----------
    left_pt, right_pt:
        Puntos ``(x, y)`` en píxeles. **Se_reordered por ``x`` internamente**
        para garantizar ``x_der > x_izq`` y así conservar el signo físico del
        resultado (izquierda/derecha en pantalla).
    """
    x1, y1 = left_pt
    x2, y2 = right_pt
    if x2 < x1:  # normaliza el orden para que dx sea siempre positivo
        x1, y1, x2, y2 = x2, y2, x1, y1
    return wrap_line_angle(degrees(math.atan2(y2 - y1, x2 - x1)))


def point_angle_deg(origin_pt: Point2, target_pt: Point2) -> float:
    """Ángulo del vector ``origin -> target`` en grados, en ``(-180, 180]``.

    A diferencia de :func:`line_angle_deg` aquí **sí** se conserva el signo,
    porque el vector tiene principio y fin.
    """
    return wrap_degrees(degrees(math.atan2(target_pt[1] - origin_pt[1],
                                          target_pt[0] - origin_pt[0])))


def signed_delta_deg(a: float, b: float) -> float:
    """Diferencia con signo ``a - b`` envuelta a ``(-180, 180]``."""
    return wrap_degrees(a - b)


# --- Filtrado temporal -------------------------------------------------------


class OneEuroFilter:
    """Filtro One Euro (Casiez, Roussel & Vogel, CHI 2012).

    Es la opción correcta para suavizar las measurements de un detector de
    landmarks: Unlike a simple moving average, it is *adaptive*, es decir,
    cuando la cabeza está quieta el suavizado es aggressive (elimina el temblor
    de la detección) y cuando se mueve el suavizado se relaja (no añade retardo
    perceptible). Ese compromiso es justo lo que necesita un monitor de
    postura en tiempo real.

    Parámetros
    ----------
    min_cutoff:
        Frecuencia de corte mínima (Hz). Valores bajos => más suavizado.
    beta:
        Ganancia de adaptatividad (coeficiente de velocidad). Valores altos
        siguen mejor los movimientos rápidos.
    d_cutoff:
        Frecuencia de corte del filtro que estima la velocidad.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        if min_cutoff <= 0:
            raise ValueError("min_cutoff debe ser > 0")
        if d_cutoff <= 0:
            raise ValueError("d_cutoff debe ser > 0")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self._x_hat: float | None = None      # valor filtrado
        self._dx_hat: float = 0.0             # velocidad filtrada
        self._last_dt: float | None = None

    # -- interno ------------------------------------------------------------
    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        """Coeficiente del filtro de primer orden para un paso de tiempo dado."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    # -- API ----------------------------------------------------------------
    def reset(self) -> None:
        """Olvida el historial (úsalo tras una pérdida de tracking)."""
        self._x_hat = None
        self._dx_hat = 0.0
        self._last_dt = None

    @property
    def initialized(self) -> bool:
        return self._x_hat is not None

    def filter(self, value: float, dt: float | None = None) -> float:
        """Filtra ``value``.

        ``dt`` es el tiempo transcurrido en segundos. Si es ``None`` se reutiliza
        el ``dt`` de la llamada anterior (o 1/30 s si es la primera).
        """
        if dt is None:
            dt = self._last_dt if self._last_dt else 1.0 / 30.0
        dt = max(float(dt), 1e-6)          # evita división por cero
        self._last_dt = dt

        if self._x_hat is None:            # primera muestra: sin histórico
            self._x_hat = float(value)
            return self._x_hat

        # 1) Estimación de velocidad (velocidad cruda -> filtrada).
        dx_raw = (value - self._x_hat) / dt
        self._dx_hat += self._alpha(self.d_cutoff, dt) * (dx_raw - self._dx_hat)

        # 2) Frecuencia de corte adaptativa: más suave si va lento.
        cutoff = self.min_cutoff + self.beta * abs(self._dx_hat)

        # 3) Suavizado de la señal.
        self._x_hat += self._alpha(cutoff, dt) * (value - self._x_hat)
        return self._x_hat


class AngleTracker:
    """Envoltorio cómodo de :class:`OneEuroFilter` para una serie de ángulos.

    Expone además ``deviation`` (respecto a un cero calibrado) para no tener
    que calibrar dos veces la misma serie en cada módulo.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self._filter = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def update(self, value: float, dt: float | None = None) -> float:
        return self._filter.filter(value, dt)

    def deviation(self, value: float, zero: float, dt: float | None = None) -> float:
        """Desviación con signo respecto al ángulo neutro ``zero``."""
        return self.update(value, dt) - zero

    def reset(self) -> None:
        self._filter.reset()


def median(values: Sequence[float]) -> float:
    """Mediana de una secuencia. Devuelve 0.0 si está vacía.

    Se prefiere la mediana a la media porque un solo fotograma con detección
    corrupta (un par de ojos invertido, un salto de detección) no desplaza el
    ángulo neutro calibrado.
    """
    data = sorted(values)
    n = len(data)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return float(data[mid])
    return (data[mid - 1] + data[mid]) / 2.0


def percentile(values: Sequence[float], pct: float) -> float:
    """Percentil ``pct`` (0-100) mediante interpolación lineal."""
    data = sorted(values)
    n = len(data)
    if n == 0:
        return 0.0
    if n == 1:
        return float(data[0])
    k = (n - 1) * (max(0.0, min(100.0, pct)) / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(data[int(k)])
    return float(data[lo] * (hi - k) + data[hi] * (k - lo))
