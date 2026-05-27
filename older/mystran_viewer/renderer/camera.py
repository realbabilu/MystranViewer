"""
Arcball camera for 3D navigation.
Controls:
  Left drag   → rotate
  Middle drag → pan
  Scroll      → zoom
  R           → reset
"""

import numpy as np
import pyrr


class Camera:
    def __init__(self):
        self.reset()

    def reset(self):
        self._yaw   = 30.0    # degrees
        self._pitch = 20.0
        self._distance = 5.0
        self._target  = np.zeros(3, dtype=np.float32)
        self._fov     = 45.0

        # Mouse state
        self._last_x = 0.0
        self._last_y = 0.0
        self._rotating = False
        self._panning  = False

    # -----------------------------------------------------------------------
    def on_mouse_button(self, button, action, x, y):
        """
        button: 0=left, 1=right, 2=middle
        action: 1=press, 0=release
        """
        if button == 0:
            self._rotating = (action == 1)
        elif button == 2 or button == 1:
            self._panning = (action == 1)
        self._last_x = x
        self._last_y = y

    def on_mouse_move(self, x, y):
        dx = x - self._last_x
        dy = y - self._last_y
        self._last_x = x
        self._last_y = y

        if self._rotating:
            self._yaw   += dx * 0.4
            self._pitch -= dy * 0.4
            self._pitch  = np.clip(self._pitch, -89.0, 89.0)

        if self._panning:
            # Pan in camera-right and camera-up directions
            right = self._right_vec()
            up    = self._up_vec()
            pan_scale = self._distance * 0.001
            self._target -= right * (dx * pan_scale)
            self._target += up   * (dy * pan_scale)

    def on_scroll(self, dy):
        self._distance *= (0.9 ** dy)
        self._distance = np.clip(self._distance, 0.01, 1e6)

    def fit(self, center: np.ndarray, radius: float):
        """Fit view to bounding sphere."""
        self._target   = center.astype(np.float32)
        self._distance = radius * 2.5
        self._fov      = 45.0

    # -----------------------------------------------------------------------
    def view_matrix(self) -> np.ndarray:
        eye = self._eye()
        return pyrr.matrix44.create_look_at(
            eye=eye,
            target=self._target,
            up=np.array([0, 1, 0], dtype=np.float32),
            dtype=np.float32
        )

    def proj_matrix(self, aspect: float) -> np.ndarray:
        return pyrr.matrix44.create_perspective_projection_matrix(
            fovy=self._fov,
            aspect=aspect,
            near=self._distance * 0.001,
            far=self._distance * 1000.0,
            dtype=np.float32
        )

    def mvp(self, aspect: float) -> np.ndarray:
        return pyrr.matrix44.multiply(
            self.view_matrix(),
            self.proj_matrix(aspect)
        )

    # -----------------------------------------------------------------------
    def _eye(self) -> np.ndarray:
        yaw_r   = np.radians(self._yaw)
        pitch_r = np.radians(self._pitch)
        x = self._distance * np.cos(pitch_r) * np.sin(yaw_r)
        y = self._distance * np.sin(pitch_r)
        z = self._distance * np.cos(pitch_r) * np.cos(yaw_r)
        return self._target + np.array([x, y, z], dtype=np.float32)

    def _right_vec(self) -> np.ndarray:
        yaw_r = np.radians(self._yaw)
        return np.array([np.cos(yaw_r), 0, -np.sin(yaw_r)], dtype=np.float32)

    def _up_vec(self) -> np.ndarray:
        # approximate: cross(forward, right)
        fwd = pyrr.vector.normalise(self._target - self._eye())
        rgt = self._right_vec()
        up  = np.cross(rgt, fwd)
        nrm = np.linalg.norm(up)
        return (up / nrm).astype(np.float32) if nrm > 1e-9 else np.array([0, 1, 0], dtype=np.float32)

    @property
    def yaw(self): return self._yaw
    @property
    def pitch(self): return self._pitch
    @property
    def distance(self): return self._distance
