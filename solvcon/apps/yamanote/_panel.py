# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Control widget for the trains: play or pause, the speed-up and the clock,
and the switches of the streets, their lane marks and their signals.

The panel is passive: it reports what its controls hold through callbacks
and shows what it is told, and the clock belongs to the controller in
:mod:`._control`.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QWidget, QFormLayout, QLabel, QComboBox,
                               QPushButton, QCheckBox)

__all__ = [  # noqa: F822
    'TrainPanel',
]


class TrainPanel(QWidget):
    """Play/Pause, the speed factor, the simulated time of day and whether
    the streets, their lane marks and crosswalks, and their signals show.

    :ivar play_toggled: Owner-supplied callback taking True to play and
        False to pause, or None.
    :ivar factor_changed: Owner-supplied callback taking the new speed
        factor, or None.
    :ivar roads_toggled: Owner-supplied callback taking True to show the
        streets and False to hide them, or None.
    :ivar marks_toggled: The same for the lane marks and crosswalks.
    :ivar signals_toggled: The same for the traffic signals.
    """

    FACTORS = (1.0, 10.0, 60.0)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.play_toggled = None
        self.factor_changed = None
        self.roads_toggled = None
        self.marks_toggled = None
        self.signals_toggled = None

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
        self._roads = QCheckBox("Roads")
        self._roads.setChecked(True)
        self._roads.toggled.connect(self._on_roads)
        self._marks = QCheckBox("Lane marks")
        self._marks.setChecked(True)
        self._marks.toggled.connect(self._on_marks)
        self._signals = QCheckBox("Signals")
        self._signals.setChecked(True)
        self._signals.toggled.connect(self._on_signals)

        form = QFormLayout(self)
        form.addRow(self._play)
        form.addRow("Speed", self._factor)
        form.addRow("Time", self._clock)
        form.addRow(self._roads)
        form.addRow(self._marks)
        form.addRow(self._signals)

    def factor(self):
        return self._factor.currentData()

    def clock_text(self):
        return self._clock.text()

    def roads_shown(self):
        return self._roads.isChecked()

    def marks_shown(self):
        return self._marks.isChecked()

    def signals_shown(self):
        return self._signals.isChecked()

    def set_roads_available(self, available):
        """Enable the street switches only when there are streets to
        show."""
        for box in (self._roads, self._marks, self._signals):
            box.setEnabled(available)

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

    def _on_roads(self, checked):
        if self.roads_toggled is not None:
            self.roads_toggled(checked)

    def _on_marks(self, checked):
        if self.marks_toggled is not None:
            self.marks_toggled(checked)

    def _on_signals(self, checked):
        if self.signals_toggled is not None:
            self.signals_toggled(checked)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
