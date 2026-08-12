"""recipe_editor — grouped editable controls over a fitted FilmModel (the edit layer).

@context  The recipe is inspectable AND editable (ADR-0002 b2.3, spec §9): after Learn, the
          fitted values become the starting point the user can hand-tweak — the Film-Stock-
          panel idea reborn on top of fitted parameters. Widgets are generated from one spec
          table over the serialize.py dict layout, so model <-> UI stays mechanical.
@done     RecipeEditor: crosstalk / per-channel curves / sat-vs-luma / hue-zone groups in a
          scroll area; set_model / model(); `edited` signal per change (debounce upstream).
@todo     Reset-to-fitted button; per-group collapse (nice-to-have).
@limits   GUI-only (Qt); no color math. Display units: hue shift in DEGREES (stored radians),
          sat trim in % (stored fraction). Spinboxes carry 4 decimals — a set_model/model()
          round-trip is exact to 1e-4 (fitted values beyond that are display-rounded).
          Ranges mirror the fit bounds so hand edits can't produce a degenerate model.
@affects  Embedded in apply_tab.py (replaces the read-only summary there). Uses
          filmmodel/serialize.py dicts. ADR-0002 b2.3.
"""

from __future__ import annotations

import math

from PySide6 import QtCore, QtWidgets

from lutgen.fitter.filmmodel import FilmModel
from lutgen.fitter.filmmodel.serialize import model_from_dict, model_to_dict

# One row per control: (section title, dict path, label, lo, hi, step, to_ui, from_ui).
# to_ui/from_ui convert stored value <-> displayed value (degrees, percent).
_ID = (lambda v: v, lambda v: v)
_DEG = (math.degrees, math.radians)
_PCT = (lambda v: v * 100.0, lambda v: v / 100.0)

_SPEC: list[tuple[str, tuple[str, ...], str, float, float, float, tuple]] = []


def _curve_rows(ch: str) -> list:
    p = ("curves", ch)
    label = ch.upper()
    return [
        (f"Tone curve {label}", (*p, "toe"), "Toe", 0.0, 2.0, 0.01, _ID),
        (f"Tone curve {label}", (*p, "shoulder"), "Shoulder", 0.0, 2.0, 0.01, _ID),
        (f"Tone curve {label}", (*p, "slope"), "Slope", 0.5, 2.0, 0.01, _ID),
        (f"Tone curve {label}", (*p, "pivot"), "Pivot", 0.3, 0.7, 0.01, _ID),
    ]


_SPEC += [("Global", ("global", "exposure"), "Exposure (DI offset)", -0.3, 0.3, 0.005, _ID)]
for _ch in ("r", "g", "b"):
    _SPEC += _curve_rows(_ch)
_SPEC += [
    ("Crosstalk", ("crosstalk", k), lbl, -0.25, 0.25, 0.005, _ID)
    for k, lbl in (("rg", "R → G"), ("rb", "R → B"), ("gr", "G → R"),
                   ("gb", "G → B"), ("br", "B → R"), ("bg", "B → G"))
]
_SPEC += [
    ("Saturation vs luminance", ("sat_luma", k), lbl, 0.7, 1.3, 0.01, _ID)
    for k, lbl in (("shadow", "Shadow ×"), ("mid", "Mid ×"), ("high", "Highlight ×"))
]
for _z in ("r", "y", "g", "c", "b", "m"):
    _SPEC += [
        ("Hue zones (legacy)", ("hue_zones", f"{_z}_shift"), f"{_z.upper()} hue °", -20.0, 20.0, 0.5, _DEG),
        ("Hue zones (legacy)", ("hue_zones", f"{_z}_trim"), f"{_z.upper()} sat %", -50.0, 50.0, 1.0, _PCT),
    ]
# v2.1 Fourier hue curve coefficients (power-user; shift in radians, trim as fraction)
for _cf in ("s0", "sc1", "sc2", "sc3", "sc4", "ss1", "ss2", "ss3", "ss4"):
    _SPEC += [("Hue curve — shift coefs", ("hue_fourier", _cf), _cf, -0.12, 0.12, 0.005, _ID)]
for _cf in ("t0", "tc1", "tc2", "tc3", "tc4", "ts1", "ts2", "ts3", "ts4"):
    _SPEC += [("Hue curve — sat coefs", ("hue_fourier", _cf), _cf, -0.25, 0.25, 0.005, _ID)]
for _cf in ("l0", "lc1", "lc2", "ls1", "ls2"):
    _SPEC += [("Hue curve — brightness mod", ("hue_fourier", _cf), _cf, -0.2, 0.2, 0.005, _ID)]
# Block F film system (ADR-0008): negative → coupling → print
_SPEC += [
    ("Film system — negative", ("film_system", "negative", "g_r"), "R gamma ×", 0.5, 2.0, 0.005, _ID),
    ("Film system — negative", ("film_system", "negative", "g_g"), "G gamma ×", 0.5, 2.0, 0.005, _ID),
    ("Film system — negative", ("film_system", "negative", "g_b"), "B gamma ×", 0.5, 2.0, 0.005, _ID),
    ("Film system — negative", ("film_system", "negative", "toe"), "Toe", 0.0, 1.0, 0.01, _ID),
    ("Film system — negative", ("film_system", "negative", "toe_at"), "Toe at (stops)", -6.0, -1.0, 0.1, _ID),
]
_SPEC += [
    ("Film system — coupling", ("film_system", "coupling", k), lbl, 0.0, 0.25, 0.005, _ID)
    for k, lbl in (("rg", "R ⊣ G"), ("rb", "R ⊣ B"), ("gr", "G ⊣ R"),
                   ("gb", "G ⊣ B"), ("br", "B ⊣ R"), ("bg", "B ⊣ G"))
]
_SPEC += [
    ("Film system — print", ("film_system", "printer", "slope"), "Contrast ×", 0.5, 2.5, 0.01, _ID),
    ("Film system — print", ("film_system", "printer", "shoulder"), "Shoulder", 0.0, 1.0, 0.01, _ID),
    ("Film system — print", ("film_system", "printer", "ptoe"), "Black convergence", 0.0, 1.0, 0.01, _ID),
    ("Film system — print", ("film_system", "printer", "range_hi"), "White at (stops)", 1.0, 6.0, 0.1, _ID),
    ("Film system — print", ("film_system", "printer", "range_lo"), "Black at (stops)", -8.0, -1.0, 0.1, _ID),
]


def _get(d: dict, path: tuple[str, ...]) -> float:
    for k in path:
        d = d[k]
    return float(d)


def _set(d: dict, path: tuple[str, ...], value: float) -> None:
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value


class RecipeEditor(QtWidgets.QScrollArea):
    """Editable grouped view of a FilmModel. Emits ``edited`` on any user change."""

    edited = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self._spins: dict[tuple[str, ...], QtWidgets.QDoubleSpinBox] = {}
        self._loading = False

        body = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(body)
        forms: dict[str, QtWidgets.QFormLayout] = {}
        for section, path, label, lo, hi, step, conv in _SPEC:
            if section not in forms:
                group = QtWidgets.QGroupBox(section)
                form = QtWidgets.QFormLayout(group)
                lay.addWidget(group)
                forms[section] = form
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setDecimals(4)
            spin.valueChanged.connect(self._on_change)
            forms[section].addRow(label, spin)
            self._spins[path] = spin
        lay.addStretch(1)

        self.setWidget(body)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.set_model(FilmModel.identity())

    def _conv_for(self, path: tuple[str, ...]) -> tuple:
        for _section, p, _label, _lo, _hi, _step, conv in _SPEC:
            if p == path:
                return conv
        raise KeyError(path)

    def set_model(self, model: FilmModel) -> None:
        """Populate the controls from ``model`` without firing ``edited``."""
        d = model_to_dict(model)
        self._loading = True
        try:
            for path, spin in self._spins.items():
                to_ui, _ = self._conv_for(path)
                spin.setValue(to_ui(_get(d, path)))
        finally:
            self._loading = False

    def model(self) -> FilmModel:
        """The model as currently edited (display-rounded to the spinbox precision)."""
        d = model_to_dict(FilmModel.identity())
        for path, spin in self._spins.items():
            _, from_ui = self._conv_for(path)
            _set(d, path, from_ui(spin.value()))
        return model_from_dict(d)

    def _on_change(self, _=None) -> None:
        if not self._loading:
            self.edited.emit()
