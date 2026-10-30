# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Control widget for the trains: play or pause, the speed-up and the clock.

The panel is passive: it reports what its controls hold through callbacks
and shows what it is told, and the clock belongs to the controller in
:mod:`._control`.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QWidget, QFormLayout, QLabel, QComboBox,
                               QPushButton)

__all__ = [  # noqa: F822
    'TrainPanel',
]


class TrainPanel(QWidget):
    """Play/Pause, the speed factor and the simulated time of day.

    :ivar play_toggled: Owner-supplied callback taking True to play and
        False to pause, or None.
    :ivar factor_changed: Owner-supplied callback taking the new speed
        factor, or None.
    """

    FACTORS = (1.0, 10.0, 60.0)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.play_toggled = None
        self.factor_changed = None

        self._play = QPushButton("Play")
        self._play.setCheckable(True)
        self._play.toggled.connect(self._on_play)
        self._factor = QComboBox()
        for factor in self.FACTORS:
            self._factor.addItem(f"{factor:g}x", factor)
        self._factor.currentIndexChanged.connect(self._on_factor)
        self._clock = QLabel("")
        self._clock.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._clock.setFont(
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))

        form = QFormLayout(self)
        form.addRow(self._play)
        form.addRow("Speed", self._factor)
        form.addRow("Time", self._clock)

    def factor(self):
        return self._factor.currentData()

    def clock_text(self):
        return self._clock.text()

    def set_playing(self, playing):
        """Reflect the play state in the button without re-firing it."""
        self._play.blockSignals(True)
        self._play.setChecked(playing)
        self._play.blockSignals(False)
        self._play.setText("Pause" if playing else "Play")

    def show_clock(self, seconds):
        """Show a time of day given in seconds after midnight."""
        minutes, sec = divmod(int(seconds) % 86400, 60)
        self._clock.setText(f"{minutes // 60:02d}:{minutes % 60:02d}:"
                            f"{sec:02d}")

    def _on_play(self, checked):
        self._play.setText("Pause" if checked else "Play")
        if self.play_toggled is not None:
            self.play_toggled(checked)

    def _on_factor(self, _):
        if self.factor_changed is not None:
            self.factor_changed(self.factor())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
