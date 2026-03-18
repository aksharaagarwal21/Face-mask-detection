# pyre-unsafe
"""
alert_system.py — Audio + visual alert system for mask violations
"""

import time
import logging
import threading
from config import (
    ALERT_ENABLED, ALERT_COOLDOWN_FRAMES, ALERT_TRIGGER_FRAMES,
    ALERT_BEEP_FREQ, ALERT_BEEP_DURATION
)

logger = logging.getLogger("AlertSystem")


class AlertSystem:
    """
    Fires audio and/or visual alerts when mask violations are detected.

    - Requires N consecutive violation frames before triggering
    - Enforces a cooldown between alerts to avoid spam
    - Runs audio in a background thread (non-blocking)
    """

    def __init__(self,
                 enabled=ALERT_ENABLED,
                 trigger_frames=ALERT_TRIGGER_FRAMES,
                 cooldown_frames=ALERT_COOLDOWN_FRAMES):
        self.enabled = enabled
        self.trigger_frames = trigger_frames
        self.cooldown_frames = cooldown_frames

        self._consecutive_violations = 0
        self._cooldown_counter = 0
        self._alert_active = False
        self._lock = threading.Lock()

        logger.info(
            f"AlertSystem init | enabled={enabled} "
            f"trigger={trigger_frames}f cooldown={cooldown_frames}f"
        )

    def update(self, has_violation: bool) -> bool:
        """
        Called each frame. Returns True if alert should be shown this frame.

        Args:
            has_violation: True if any violation detected in this frame

        Returns:
            bool: whether to display alert banner this frame
        """
        if not self.enabled:
            return False

        with self._lock:
            # Decrement cooldown
            if self._cooldown_counter > 0:
                self._cooldown_counter -= 1

            if has_violation:
                self._consecutive_violations += 1
            else:
                self._consecutive_violations = 0
                self._alert_active = False
                return False

            # Trigger alert if enough consecutive frames and cooldown expired
            if (self._consecutive_violations >= self.trigger_frames
                    and self._cooldown_counter == 0):
                self._alert_active = True
                self._cooldown_counter = self.cooldown_frames
                self._fire_beep()
                return True

            return self._alert_active

    def _fire_beep(self):
        """Play system beep in a background thread to avoid blocking the camera loop."""
        def _beep():
            try:
                import winsound
                winsound.Beep(ALERT_BEEP_FREQ, ALERT_BEEP_DURATION)  # pyre-ignore[16]
            except ImportError:
                # Linux/Mac fallback — print ASCII bell
                try:
                    import os
                    os.system('printf "\a"')
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"Beep error: {e}")

        threading.Thread(target=_beep, daemon=True).start()

    def reset(self):
        """Reset alert state (call at start of new session)."""
        with self._lock:
            self._consecutive_violations = 0
            self._cooldown_counter = 0
            self._alert_active = False

    @property
    def is_active(self):
        return self._alert_active

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        logger.info(f"Alert system {'enabled' if enabled else 'disabled'}")
