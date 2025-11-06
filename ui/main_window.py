"""Main window for the LUX koordinat kontrol arayüzü."""
from __future__ import annotations

import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2  # type: ignore[import]
import numpy as np  # type: ignore[import]
try:
    import pyautogui  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pyautogui = None  # type: ignore[assignment]
from PySide6.QtCore import QFile, QProcess, QTimer, QSignalBlocker
from PySide6.QtGui import QCloseEvent
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QWidget,
)

from priest.cure_micro import CureConfig, CureMicroController
from priest.undy_micro import UndyConfig, UndyMicroController
from priest.ac_micro import AcConfig, AcMicroController
from priest.str30_micro import Str30Config, Str30MicroController
from priest.poison_cure_micro import PoisonCureConfig, PoisonCureMicroController
from priest.rsiz_atak_micro import RsizAtakConfig, RsizAtakMicroController
from priest.rr_skill_micro import RrSkillConfig, RrSkillMicroController
from priest.parazit_micro import ParazitConfig, ParazitMicroController
from priest.malice_micro import MaliceConfig, MaliceMicroController
from priest.tekli_atak_kirma_micro import TekliAtakKirmaConfig, TekliAtakKirmaMicroController
from priest.torment_micro import TormentConfig, TormentMicroController
from priest.subside_micro import SubsideConfig, SubsideMicroController
from priest.restore_micro import RestoreConfig, RestoreMicroController
from priest.toplu10k_micro import Toplu10kConfig, Toplu10kMicroController
from priest.toplu_buff_micro import TopluBuffConfig, TopluBuffMicroController
from priest.toplu_ac_micro import TopluAcConfig, TopluAcMicroController
from priest.mana_micro import ManaConfig, ManaMicroController
from priest.heal_micro import HealConfig, HealMicroController
from priest.toplu_cure_micro import TopluCureConfig, TopluCureMicroController
from .styles import BASE_QSS
from common import monster as monster_detector

UI_PATH = Path(__file__).with_name("main_window.ui")
PROJECT_ROOT = UI_PATH.parent.parent
CM_TO_PX = 37.7952755906

TURKISH_Q_KEYS: List[str] = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "0",
    "q",
    "w",
    "e",
    "r",
    "t",
    "y",
    "u",
    "ı",
    "o",
    "p",
    "ğ",
    "ü",
    "a",
    "s",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "ş",
    "i",
    "z",
    "x",
    "c",
    "v",
    "b",
    "n",
    "m",
    "ö",
    "ç",
    ",",
    ".",
    "-",
    "=",
]
FUNCTION_SHORTCUT_KEYS: List[str] = [f"f{i}" for i in range(1, 13)]
AVAILABLE_SHORTCUT_KEYS: List[str] = TURKISH_Q_KEYS + ["space"] + FUNCTION_SHORTCUT_KEYS
DEFAULT_PARTY_SIZE = 8
HP_THRESHOLD_VALUES: List[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]


@dataclass
class SkillRow:
    """Row storing checkbox with primary/function combo boxes."""

    checkbox: QCheckBox
    label: QLabel
    primary_combo: QComboBox
    function_combo: QComboBox


class MainWindow(QMainWindow):
    """Primary window that follows the örnekgörsel layout with skill checklists."""

    def __init__(self) -> None:
        super().__init__()
        if not UI_PATH.exists():
            raise FileNotFoundError(f"UI dosyası bulunamadı: {UI_PATH}")

        self._load_ui()
        self.apply_styles(BASE_QSS)

        self._jobs: List[str] = ["Priest", "PartyMs"]

        # Cache structural widgets
        self.job_stack: Optional[QStackedWidget] = self.findChild(QStackedWidget, "JobStackedWidget")  # type: ignore[assignment]

        # Collect navigation buttons
        self._job_buttons: Dict[str, QAbstractButton] = {}
        self._job_button_group = QButtonGroup(self)
        self._job_button_group.setExclusive(True)

        for index, job in enumerate(self._jobs):
            button = self.findChild(QAbstractButton, f"Nav{job}Button")
            if button is None:
                continue
            button.setCheckable(True)
            self._job_button_group.addButton(button, index)
            self._job_buttons[job] = button

        self._job_button_group.idClicked.connect(self._on_job_selected)

        # Collect skill rows per job page
        self._skill_rows: Dict[str, List[SkillRow]] = self._gather_skill_rows()

        if pyautogui is not None:
            pyautogui.FAILSAFE = False
        else:
            self._show_status("pyautogui bulunamadı; tıklama devre dışı.")
        try:
            self._anti_afk_template = monster_detector.load_template()
        except FileNotFoundError as exc:
            self._anti_afk_template = None
            self._show_status(str(exc))
        self._party_offset_combos: Dict[int, QComboBox] = {}

        if self.job_stack:
            self.job_stack.setCurrentIndex(0)
        default_button = self._job_buttons.get("Priest")
        if default_button:
            default_button.setChecked(True)

        self._party_coordinates: Dict[int, Tuple[int, int]] = {}
        self._navigation_queue: List[int] = []
        self._current_navigation_row: Optional[int] = None
        self._navigation_process: Optional[QProcess] = None
        self._monitor_after_completion = False
        self._paused = False
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._check_current_target_status)
        self._status_checks = 0
        self._pending_auto_click_row: Optional[int] = None

        # Canavar durum takibi
        self._monitor_delay_timer = QTimer(self)
        self._monitor_delay_timer.setSingleShot(True)
        self._monitor_delay_timer.timeout.connect(self._begin_monitor_checks)
        self._monitor_timer = QTimer(self)
        self._monitor_timer.setInterval(5000)
        self._monitor_timer.timeout.connect(self._monitor_tick)
        self._monitor_row: Optional[int] = None
        self._monitor_failures = 0
        self._monitor_should_advance = False
        self._monitor_label: Optional[QLabel] = None

        self._cure_controller: Optional[CureMicroController] = None
        self._global_start_checkbox: Optional[QCheckBox] = None
        self._cure_auto_checkbox: Optional[QCheckBox] = None
        self._cure_shortcut_checkbox: Optional[QCheckBox] = None
        self._cure_shortcut_combo: Optional[QComboBox] = None
        self._cure_row: Optional[SkillRow] = None
        self._priest_shortcut_assignments: Dict[str, str] = {}
        self._shortcut_selected_keys: Dict[str, Optional[str]] = {}
        self._zehir_shortcut_combo: Optional[QComboBox] = None
        self._zehir_row: Optional[SkillRow] = None
        self._rsiz_shortcut_combo: Optional[QComboBox] = None
        self._rsiz_row: Optional[SkillRow] = None
        self._parazit_controller: Optional[ParazitMicroController] = None
        self._parazit_row: Optional[SkillRow] = None
        self._parazit_timer_checkbox: Optional[QCheckBox] = None
        self._parazit_partyms_checkbox: Optional[QCheckBox] = None
        self._parazit_interval_spin: Optional[QSpinBox] = None
        self._malice_controller: Optional[MaliceMicroController] = None
        self._malice_row: Optional[SkillRow] = None
        self._malice_timer_checkbox: Optional[QCheckBox] = None
        self._malice_partyms_checkbox: Optional[QCheckBox] = None
        self._malice_interval_spin: Optional[QSpinBox] = None
        self._malice_registered_with_parazit = False
        self._tekli_controller: Optional[TekliAtakKirmaMicroController] = None
        self._tekli_row: Optional[SkillRow] = None
        self._tekli_timer_checkbox: Optional[QCheckBox] = None
        self._tekli_partyms_checkbox: Optional[QCheckBox] = None
        self._tekli_interval_spin: Optional[QSpinBox] = None
        self._tekli_registered_with_malice = False
        self._subside_controller: Optional[SubsideMicroController] = None
        self._subside_row: Optional[SkillRow] = None
        self._subside_interval_spin: Optional[QSpinBox] = None
        self._torment_controller: Optional[TormentMicroController] = None
        self._torment_row: Optional[SkillRow] = None
        self._torment_interval_spin: Optional[QSpinBox] = None
        self._rr_controller: Optional[RrSkillMicroController] = None
        self._rr_row: Optional[SkillRow] = None
        self._rr_shortcut_combo: Optional[QComboBox] = None
        self._toplu_ac_controller: Optional[TopluAcMicroController] = None
        self._toplu_ac_master_checkbox: Optional[QCheckBox] = None
        self._toplu_ac_primary_combo: Optional[QComboBox] = None
        self._toplu_ac_function_combo: Optional[QComboBox] = None
        self._toplu_ac_precure_checkbox: Optional[QCheckBox] = None
        self._toplu_cure_controller: Optional[TopluCureMicroController] = None
        self._toplu_cure_row: Optional[SkillRow] = None
        self._party_shared_calibrate_button: Optional[QPushButton] = None
        self._party_shared_party_size_spin: Optional[QSpinBox] = None
        self._party_shared_threshold_combo: Optional[QComboBox] = None
        self._party_shared_calibration_running: bool = False
        self._toplu_controller: Optional[Toplu10kMicroController] = None
        self._toplu_row: Optional[SkillRow] = None
        self._toplu_buff_controller: Optional[TopluBuffMicroController] = None
        self._toplu_buff_row: Optional[SkillRow] = None
        self._toplu_buff_precure_checkbox: Optional[QCheckBox] = None
        self._restore_controller: Optional[RestoreMicroController] = None
        self._restore_master_checkbox: Optional[QCheckBox] = None
        self._restore_primary_combo: Optional[QComboBox] = None
        self._restore_function_combo: Optional[QComboBox] = None

        self._initialize_cure_controls()
        self._undy_controller: Optional[UndyMicroController] = None
        self._undy_row: Optional[SkillRow] = None
        self._initialize_undy_controls()
        self._ac_controller: Optional[AcMicroController] = None
        self._ac_row: Optional[SkillRow] = None
        self._initialize_ac_controls()
        self._mana_controller: Optional[ManaMicroController] = None
        self._mana_row: Optional[SkillRow] = None
        self._mana_threshold_combo: Optional[QComboBox] = None
        self._initialize_mana_controls()
        self._heal_controller: Optional[HealMicroController] = None
        self._heal_row: Optional[SkillRow] = None
        self._heal_threshold_combo: Optional[QComboBox] = None
        self._heal_last_hint: Optional[str] = None
        self._initialize_heal_controls()
        self._str30_controller: Optional[Str30MicroController] = None
        self._str30_row: Optional[SkillRow] = None
        self._initialize_str30_controls()
        self._zehir_controller: Optional[PoisonCureMicroController] = None
        self._initialize_zehir_controls()
        self._rsiz_controller: Optional[RsizAtakMicroController] = None
        self._initialize_rsiz_atak_controls()
        self._initialize_parazit_controls()
        self._rr_controller: Optional[RrSkillMicroController] = None
        self._initialize_rr_skill_controls()
        self._initialize_malice_controls()
        self._initialize_tekli_controls()
        self._initialize_subside_controls()
        self._initialize_torment_controls()
        self._initialize_party_shared_controls()
        self._initialize_toplu_ac_controls()
        self._initialize_toplu_cure_controls()
        self._initialize_restore_controls()
        self._initialize_toplu_buff_controls()
        self._initialize_toplu_controls()

        self._wire_party_ms_actions()

    def apply_styles(self, qss: str) -> None:
        """Apply the stylesheet to the window."""
        self.setStyleSheet(qss)

    def _load_ui(self) -> None:
        """Load the .ui file via QUiLoader and reparent widgets."""
        ui_file = QFile(str(UI_PATH))
        if not ui_file.open(QFile.ReadOnly):
            raise OSError(f"UI dosyası açılamadı: {UI_PATH}")

        loader = QUiLoader()
        try:
            loaded = loader.load(ui_file, None)
        finally:
            ui_file.close()

        if loaded is None:
            raise ValueError("UI yüklenemedi.")

        if isinstance(loaded, QMainWindow):
            self.setWindowTitle(loaded.windowTitle())
            self.resize(loaded.size())

            central_widget = loaded.centralWidget()
            if central_widget:
                central_widget.setParent(self)
                loaded.setCentralWidget(None)
                self.setCentralWidget(central_widget)
            status_bar = loaded.statusBar()
            if status_bar:
                status_bar.setParent(self)
                loaded.setStatusBar(None)
                self.setStatusBar(status_bar)
            menu_bar = loaded.menuBar()
            if menu_bar:
                menu_bar.setParent(self)
                loaded.setMenuBar(None)
                self.setMenuBar(menu_bar)
            self._loaded_ui = loaded
        elif isinstance(loaded, QWidget):
            self.setCentralWidget(loaded)
            self._loaded_ui = loaded
        else:
            raise TypeError("Beklenmeyen UI kök türü.")

    def _gather_skill_rows(self) -> Dict[str, List[SkillRow]]:
        """Collect support, attack, and party skill rows for the priest layout."""
        skills: Dict[str, List[SkillRow]] = {"Priest": []}

        def collect(prefix: str) -> None:
            index = 1
            while True:
                checkbox = self.findChild(QCheckBox, f"{prefix}{index}CheckBox")
                label = self.findChild(QLabel, f"{prefix}{index}Label")
                primary = self.findChild(QComboBox, f"{prefix}{index}PrimaryComboBox")
                function = self.findChild(QComboBox, f"{prefix}{index}FunctionComboBox")
                if None in (checkbox, label, primary, function):
                    break
                skills["Priest"].append(SkillRow(checkbox, label, primary, function))
                index += 1

        collect("PriestSupport")
        collect("PriestAttack")
        collect("PriestPartyRestore")
        collect("PriestPartyToplu")
        return skills

    def _wire_party_ms_actions(self) -> None:
        """Connect Party MS calibration, save, and navigation buttons."""
        calibrate_button = self.findChild(QPushButton, "PartyMsCalibrateHeaderButton")
        if calibrate_button:

            def trigger_calibration() -> None:
                QProcess.startDetached(
                    sys.executable,
                    ["-m", "common.read_coordinates", "--calibrate"],
                    str(PROJECT_ROOT),
                )

            calibrate_button.clicked.connect(trigger_calibration)

        for row_index in range(1, 18):
            coord_label = self.findChild(QLabel, f"PartyMsRow{row_index}CoordLabel")
            offset_combo = self.findChild(QComboBox, f"PartyMsRow{row_index}OffsetComboBox")
            current_button = self.findChild(QPushButton, f"PartyMsRow{row_index}CurrentButton")
            target_button = self.findChild(QPushButton, f"PartyMsRow{row_index}TargetButton")

            if coord_label:
                coord_label.setText("--")

            if offset_combo:
                self._party_offset_combos[row_index] = offset_combo
                offset_combo.setCurrentText("3")

            if current_button and coord_label:
                current_button.clicked.connect(self._make_coordinate_saver(row_index, coord_label))

            if target_button:
                target_button.clicked.connect(
                    lambda _=False, r=row_index: self._launch_direct_navigation(r)
                )

        start_button = self.findChild(QPushButton, "PartyMsStartButton")
        if start_button:
            start_button.setEnabled(True)
            start_button.setToolTip("Kaydedilen koordinatlara sırayla git.")
            start_button.clicked.connect(self._start_navigation_sequence)

        pause_button = self.findChild(QPushButton, "PartyMsPauseButton")
        if pause_button:
            pause_button.setEnabled(True)
            pause_button.setToolTip("Aktif rotayı duraklat.")
            pause_button.clicked.connect(self._pause_navigation)

        stop_button = self.findChild(QPushButton, "PartyMsStopButton")
        if stop_button:
            stop_button.setEnabled(True)
            stop_button.setToolTip("Tüm navigasyonu sonlandır.")
            stop_button.clicked.connect(self._stop_navigation)

        reset_button = self.findChild(QPushButton, "PartyMsResetButton")
        if reset_button:
            reset_button.setEnabled(True)
            reset_button.setToolTip("Kaydedilen koordinatları temizle.")
            reset_button.clicked.connect(self._reset_navigation)

    def _initialize_cure_controls(self) -> None:
        """Setup Cure-specific UI controls and controller."""
        self._global_start_checkbox = self.findChild(QCheckBox, "PriestGlobalStartCheckBox")
        self._cure_auto_checkbox = self.findChild(QCheckBox, "PriestSupport1AutoMiniCheckBox")
        self._cure_shortcut_checkbox = self.findChild(QCheckBox, "PriestSupport1ShortcutMiniCheckBox")
        self._cure_shortcut_combo = self.findChild(QComboBox, "PriestSupport1ShortcutComboBox")
        self._cure_row = self._find_cure_row()

        status_callback = self._make_threadsafe_status_callback()
        try:
            self._cure_controller = CureMicroController(status_callback=status_callback)
        except Exception as exc:  # pragma: no cover - defensive guard for optional deps
            self._cure_controller = None
            self._show_status(f"Cure mikro başlatılamadı: {exc}", 6000)
            print(f"Cure controller could not be initialized: {exc}", file=sys.stderr)

        combo = self._cure_shortcut_combo
        if combo is not None:
            combo.setEnabled(False)
            self._ensure_shortcut_combo(combo)

        if self._global_start_checkbox is not None:
            self._global_start_checkbox.toggled.connect(lambda _=False: self._sync_cure_configuration())
        if self._cure_row is not None:
            self._cure_row.checkbox.toggled.connect(lambda _=False: self._sync_cure_configuration())
            self._cure_row.primary_combo.currentIndexChanged.connect(lambda _=0: self._sync_cure_configuration())
            self._cure_row.function_combo.currentIndexChanged.connect(lambda _=0: self._sync_cure_configuration())
        if self._cure_auto_checkbox is not None:
            self._cure_auto_checkbox.toggled.connect(lambda _=False: self._sync_cure_configuration())
        if self._cure_shortcut_checkbox is not None:
            self._cure_shortcut_checkbox.toggled.connect(self._on_cure_shortcut_toggled)
        if combo is not None:
            combo.currentIndexChanged.connect(self._on_cure_shortcut_key_changed)

        self._sync_cure_configuration()

    def _initialize_undy_controls(self) -> None:
        """Setup Undy-specific UI controls and controller."""
        self._undy_row = self._find_undy_row()

        status_callback = self._make_threadsafe_status_callback()
        try:
            self._undy_controller = UndyMicroController(status_callback=status_callback)
        except Exception as exc:  # pragma: no cover - optional deps
            self._undy_controller = None
            self._show_status(f"Undy mikro başlatılamadı: {exc}", 6000)
            print(f"Undy controller could not be initialized: {exc}", file=sys.stderr)

        if self._global_start_checkbox is not None:
            self._global_start_checkbox.toggled.connect(lambda _=False: self._sync_undy_configuration())
        if self._undy_row is not None:
            self._undy_row.checkbox.toggled.connect(lambda _=False: self._sync_undy_configuration())
            self._undy_row.primary_combo.currentIndexChanged.connect(lambda _=0: self._sync_undy_configuration())
            self._undy_row.function_combo.currentIndexChanged.connect(lambda _=0: self._sync_undy_configuration())
        self._sync_undy_configuration()

    def _initialize_ac_controls(self) -> None:
        """Setup AC-specific UI controls and controller."""
        self._ac_row = self._find_ac_row()

        status_callback = self._make_threadsafe_status_callback()
        try:
            self._ac_controller = AcMicroController(status_callback=status_callback)
        except Exception as exc:  # pragma: no cover - optional deps
            self._ac_controller = None
            self._show_status(f"AC mikro başlatılamadı: {exc}", 6000)
            print(f"AC controller could not be initialized: {exc}", file=sys.stderr)

        if self._global_start_checkbox is not None:
            self._global_start_checkbox.toggled.connect(lambda _=False: self._sync_ac_configuration())
        if self._ac_row is not None:
            self._ac_row.checkbox.toggled.connect(lambda _=False: self._sync_ac_configuration())
            self._ac_row.primary_combo.currentIndexChanged.connect(lambda _=0: self._sync_ac_configuration())
            self._ac_row.function_combo.currentIndexChanged.connect(lambda _=0: self._sync_ac_configuration())

        self._sync_ac_configuration()

    def _initialize_mana_controls(self) -> None:
        """Setup Mana controls and controller."""
        self._mana_row = self._find_mana_row()
        self._mana_threshold_combo = self.findChild(QComboBox, "PriestSupport8ThresholdComboBox")

        status_callback = self._make_threadsafe_status_callback()
        try:
            self._mana_controller = ManaMicroController(status_callback=status_callback)
        except Exception as exc:  # pragma: no cover - optional deps
            self._mana_controller = None
            self._show_status(f"Mana mikro başlatılamadı: {exc}", 6000)
            print(f"Mana controller could not be initialized: {exc}", file=sys.stderr)

        if self._global_start_checkbox is not None:
            self._global_start_checkbox.toggled.connect(lambda _=False: self._sync_mana_configuration())
        if self._mana_row is not None:
            self._mana_row.checkbox.toggled.connect(lambda _=False: self._sync_mana_configuration())
            self._mana_row.primary_combo.currentIndexChanged.connect(lambda _=0: self._sync_mana_configuration())
            self._mana_row.function_combo.currentIndexChanged.connect(lambda _=0: self._sync_mana_configuration())
        if self._mana_threshold_combo is not None:
            self._mana_threshold_combo.currentIndexChanged.connect(lambda _=0: self._sync_mana_configuration())

        self._sync_mana_configuration()

    def _initialize_heal_controls(self) -> None:
        """Setup Heal controls and controller."""
        self._heal_row = self._find_heal_row()
        self._heal_threshold_combo = self.findChild(QComboBox, "PriestSupport9ThresholdComboBox")

        status_callback = self._make_threadsafe_status_callback()
        try:
            self._heal_controller = HealMicroController(status_callback=status_callback)
        except Exception as exc:  # pragma: no cover - optional deps
            self._heal_controller = None
            self._show_status(f"Heal mikro başlatılamadı: {exc}", 6000)
            print(f"Heal controller could not be initialized: {exc}", file=sys.stderr)

        if self._global_start_checkbox is not None:
            self._global_start_checkbox.toggled.connect(lambda _=False: self._sync_heal_configuration())
        if self._heal_row is not None:
            self._heal_row.checkbox.toggled.connect(lambda _=False: self._sync_heal_configuration())
            self._heal_row.primary_combo.currentIndexChanged.connect(lambda _=0: self._sync_heal_configuration())
            self._heal_row.function_combo.currentIndexChanged.connect(lambda _=0: self._sync_heal_configuration())
        if self._heal_threshold_combo is not None:
            self._heal_threshold_combo.currentIndexChanged.connect(lambda _=0: self._sync_heal_configuration())

        self._sync_heal_configuration()

    def _initialize_str30_controls(self) -> None:
        """Setup STR30-specific UI controls and controller."""
        self._str30_row = self._find_str30_row()

        status_callback = self._make_threadsafe_status_callback()
        try:
            self._str30_controller = Str30MicroController(status_callback=status_callback)
        except Exception as exc:  # pragma: no cover - optional deps
            self._str30_controller = None
            self._show_status(f"STR 30 mikro başlatılamadı: {exc}", 6000)
            print(f"STR30 controller could not be initialized: {exc}", file=sys.stderr)

        if self._global_start_checkbox is not None:
            self._global_start_checkbox.toggled.connect(lambda _=False: self._sync_str30_configuration())
        if self._str30_row is not None:
            self._str30_row.checkbox.toggled.connect(lambda _=False: self._sync_str30_configuration())
            self._str30_row.primary_combo.currentIndexChanged.connect(lambda _=0: self._sync_str30_configuration())
            self._str30_row.function_combo.currentIndexChanged.connect(lambda _=0: self._sync_str30_configuration())

        self._sync_str30_configuration()

    def _initialize_zehir_controls(self) -> None:
        """Setup Zehir Cure shortcut controls and controller."""
        self._zehir_row = self._find_zehir_row()
        self._zehir_shortcut_combo = self.findChild(QComboBox, "PriestSupport6ShortcutComboBox")

        status_callback = self._make_threadsafe_status_callback()
        try:
            self._zehir_controller = PoisonCureMicroController(status_callback=status_callback)
        except Exception as exc:  # pragma: no cover - optional deps
            self._zehir_controller = None
            self._show_status(f"Zehir Cure mikro başlatılamadı: {exc}", 6000)
            print(f"Zehir Cure controller could not be initialized: {exc}", file=sys.stderr)

        combo = self._zehir_shortcut_combo
        if combo is not None:
            blocker = QSignalBlocker(combo)
            self._populate_shortcut_combo(combo)
            combo.setCurrentIndex(0)
            del blocker

        if self._global_start_checkbox is not None:
            self._global_start_checkbox.toggled.connect(lambda _=False: self._sync_zehir_configuration())
        if self._zehir_row is not None:
            self._zehir_row.checkbox.toggled.connect(lambda _=False: self._sync_zehir_configuration())
            self._zehir_row.primary_combo.currentIndexChanged.connect(lambda _=0: self._sync_zehir_configuration())
            self._zehir_row.function_combo.currentIndexChanged.connect(lambda _=0: self._sync_zehir_configuration())
        if combo is not None:
            combo.currentIndexChanged.connect(self._on_zehir_shortcut_key_changed)

        self._sync_zehir_configuration()

    def _initialize_rsiz_atak_controls(self) -> None:
        """Setup R'siz Atak shortcut controls and controller."""
        self._rsiz_row = self._find_rsiz_row()
        self._rsiz_shortcut_combo = self.findChild(QComboBox, "PriestAttack1ShortcutComboBox")

        status_callback = self._make_threadsafe_status_callback()
        try:
            self._rsiz_controller = RsizAtakMicroController(status_callback=status_callback)
        except Exception as exc:  # pragma: no cover - optional deps
            self._rsiz_controller = None
            self._show_status(f"R'siz Atak mikro başlatılamadı: {exc}", 6000)
            print(f"Rsiz Atak controller could not be initialized: {exc}", file=sys.stderr)

        combo = self._rsiz_shortcut_combo
        if combo is not None:
            blocker = QSignalBlocker(combo)
            self._populate_shortcut_combo(combo)
            combo.setCurrentIndex(0)
            del blocker

        if combo is not None:
            combo.currentIndexChanged.connect(self._on_rsiz_shortcut_changed)

        if self._global_start_checkbox is not None:
            self._global_start_checkbox.toggled.connect(lambda _=False: self._sync_rsiz_configuration())
        if self._rsiz_row is not None:
            self._rsiz_row.checkbox.toggled.connect(lambda _=False: self._sync_rsiz_configuration())
            self._rsiz_row.primary_combo.currentIndexChanged.connect(lambda _=0: self._sync_rsiz_configuration())

        self._sync_rsiz_configuration()

    def _initialize_rr_skill_controls(self) -> None:
        """Setup RR skill shortcut controls and controller."""
        self._rr_row = self._find_rr_row()
        self._rr_shortcut_combo = self.findChild(QComboBox, "PriestAttack2ShortcutComboBox")

        status_callback = self._make_threadsafe_status_callback()
        try:
            self._rr_controller = RrSkillMicroController(status_callback=status_callback)
        except Exception as exc:  # pragma: no cover - optional deps
            self._rr_controller = None
            self._show_status(f"RR-skill mikro başlatılamadı: {exc}", 6000)
            print(f"RR skill controller could not be initialized: {exc}", file=sys.stderr)

        combo = self._rr_shortcut_combo
        if combo is not None:
            blocker = QSignalBlocker(combo)
            self._populate_shortcut_combo(combo)
            combo.setCurrentIndex(0)
            del blocker
        if combo is not None:
            combo.currentIndexChanged.connect(self._on_rr_shortcut_changed)

        if self._global_start_checkbox is not None:
            self._global_start_checkbox.toggled.connect(lambda _=False: self._sync_rr_configuration())
        if self._rr_row is not None:
            self._rr_row.checkbox.toggled.connect(lambda _=False: self._sync_rr_configuration())
            self._rr_row.primary_combo.currentIndexChanged.connect(lambda _=0: self._sync_rr_configuration())

        self._sync_rr_configuration()

    def _initialize_malice_controls(self) -> None:
        """Setup Malice controls and controller."""
        self._malice_row = self._find_malice_row()
        self._malice_timer_checkbox = self.findChild(QCheckBox, "PriestAttack4TimerMiniCheckBox")
        self._malice_partyms_checkbox = self.findChild(QCheckBox, "PriestAttack4PartyMsMiniCheckBox")
        self._malice_interval_spin = self.findChild(QSpinBox, "PriestAttack4IntervalSpinBox")

        status_callback = self._make_threadsafe_status_callback()
        try:
            self._malice_controller = MaliceMicroController(status_callback=status_callback)
        except Exception as exc:  # pragma: no cover - optional deps
            self._malice_controller = None
            self._show_status(f"Malice mikro başlatılamadı: {exc}", 6000)
            print(f"Malice controller could not be initialized: {exc}", file=sys.stderr)

        if self._global_start_checkbox is not None:
            self._global_start_checkbox.toggled.connect(lambda _=False: self._sync_malice_configuration())
        if self._malice_row is not None:
            self._malice_row.checkbox.toggled.connect(lambda _=False: self._sync_malice_configuration())
            self._malice_row.primary_combo.currentIndexChanged.connect(lambda _=0: self._sync_malice_configuration())
            self._malice_row.function_combo.currentIndexChanged.connect(lambda _=0: self._sync_malice_configuration())

        if self._malice_timer_checkbox is not None:
            self._malice_timer_checkbox.toggled.connect(self._on_malice_timer_toggled)
        if self._malice_partyms_checkbox is not None:
            self._malice_partyms_checkbox.toggled.connect(self._on_malice_partyms_toggled)
        if self._malice_interval_spin is not None:
            self._malice_interval_spin.valueChanged.connect(lambda _=0: self._sync_malice_configuration())

        self._sync_malice_configuration()

    def _initialize_tekli_controls(self) -> None:
        """Setup Tekli Atak Kırma controls and controller."""
        self._tekli_row = self._find_tekli_row()
        self._tekli_timer_checkbox = self.findChild(QCheckBox, "PriestAttack5TimerMiniCheckBox")
        self._tekli_partyms_checkbox = self.findChild(QCheckBox, "PriestAttack5PartyMsMiniCheckBox")
        self._tekli_interval_spin = self.findChild(QSpinBox, "PriestAttack5IntervalSpinBox")

        status_callback = self._make_threadsafe_status_callback()
        try:
            self._tekli_controller = TekliAtakKirmaMicroController(status_callback=status_callback)
        except Exception as exc:  # pragma: no cover - optional deps
            self._tekli_controller = None
            self._show_status(f"Tekli Atak Kırma mikro başlatılamadı: {exc}", 6000)
            print(f"Tekli Atak Kırma controller could not be initialized: {exc}", file=sys.stderr)

        if self._global_start_checkbox is not None:
            self._global_start_checkbox.toggled.connect(lambda _=False: self._sync_tekli_configuration())
        if self._tekli_row is not None:
            self._tekli_row.checkbox.toggled.connect(lambda _=False: self._sync_tekli_configuration())
            self._tekli_row.primary_combo.currentIndexChanged.connect(lambda _=0: self._sync_tekli_configuration())
            self._tekli_row.function_combo.currentIndexChanged.connect(lambda _=0: self._sync_tekli_configuration())

        if self._tekli_timer_checkbox is not None:
            self._tekli_timer_checkbox.toggled.connect(self._on_tekli_timer_toggled)
        if self._tekli_partyms_checkbox is not None:
            self._tekli_partyms_checkbox.toggled.connect(self._on_tekli_partyms_toggled)
        if self._tekli_interval_spin is not None:
            self._tekli_interval_spin.valueChanged.connect(lambda _=0: self._sync_tekli_configuration())

        self._sync_tekli_configuration()

    def _initialize_subside_controls(self) -> None:
        """Setup Subside controls and controller."""
        self._subside_row = self._find_subside_row()
        self._subside_interval_spin = self.findChild(QSpinBox, "PriestAttack6IntervalSpinBox")

        status_callback = self._make_threadsafe_status_callback()
        try:
            self._subside_controller = SubsideMicroController(status_callback=status_callback)
        except Exception as exc:  # pragma: no cover - optional deps
            self._subside_controller = None
            self._show_status(f"Subside mikro başlatılamadı: {exc}", 6000)
            print(f"Subside controller could not be initialized: {exc}", file=sys.stderr)

        if self._global_start_checkbox is not None:
            self._global_start_checkbox.toggled.connect(lambda _=False: self._sync_subside_configuration())
        if self._subside_row is not None:
            self._subside_row.checkbox.toggled.connect(lambda _=False: self._sync_subside_configuration())
            self._subside_row.primary_combo.currentIndexChanged.connect(lambda _=0: self._sync_subside_configuration())
            self._subside_row.function_combo.currentIndexChanged.connect(lambda _=0: self._sync_subside_configuration())

        if self._subside_interval_spin is not None:
            self._subside_interval_spin.valueChanged.connect(lambda _=0: self._sync_subside_configuration())

        self._sync_subside_configuration()

    def _initialize_torment_controls(self) -> None:
        """Setup Torment controls and controller."""
        self._torment_row = self._find_torment_row()
        self._torment_interval_spin = self.findChild(QSpinBox, "PriestAttack7IntervalSpinBox")

        status_callback = self._make_threadsafe_status_callback()
        try:
            self._torment_controller = TormentMicroController(status_callback=status_callback)
        except Exception as exc:  # pragma: no cover - optional deps
            self._torment_controller = None
            self._show_status(f"Torment mikro başlatılamadı: {exc}", 6000)
            print(f"Torment controller could not be initialized: {exc}", file=sys.stderr)

        if self._global_start_checkbox is not None:
            self._global_start_checkbox.toggled.connect(lambda _=False: self._sync_torment_configuration())
        if self._torment_row is not None:
            self._torment_row.checkbox.toggled.connect(lambda _=False: self._sync_torment_configuration())
            self._torment_row.primary_combo.currentIndexChanged.connect(lambda _=0: self._sync_torment_configuration())
            self._torment_row.function_combo.currentIndexChanged.connect(lambda _=0: self._sync_torment_configuration())

        if self._torment_interval_spin is not None:
            self._torment_interval_spin.valueChanged.connect(lambda _=0: self._sync_torment_configuration())

        self._sync_torment_configuration()

    def _initialize_party_shared_controls(self) -> None:
        """Setup shared party HP controls used by Restore."""
        self._party_shared_calibrate_button = self.findChild(QPushButton, "PriestPartySharedCalibrateButton")
        self._party_shared_party_size_spin = self.findChild(QSpinBox, "PriestPartySharedPartySizeSpinBox")
        self._party_shared_threshold_combo = self.findChild(QComboBox, "PriestPartySharedThresholdComboBox")

        if self._party_shared_party_size_spin is not None:
            self._party_shared_party_size_spin.setRange(2, 8)
            if self._party_shared_party_size_spin.value() < 2:
                self._party_shared_party_size_spin.setValue(2)
            self._party_shared_party_size_spin.valueChanged.connect(lambda _=0: self._on_shared_party_hp_changed())

        if self._party_shared_threshold_combo is not None:
            if self._party_shared_threshold_combo.count() > 0:
                self._party_shared_threshold_combo.setCurrentIndex(0)
            self._party_shared_threshold_combo.currentIndexChanged.connect(lambda _=0: self._on_shared_party_hp_changed())

        if self._party_shared_calibrate_button is not None:
            self._party_shared_calibrate_button.clicked.connect(self._on_party_shared_calibrate_clicked)

        self._on_shared_party_hp_changed()

    def _initialize_toplu_ac_controls(self) -> None:
        """Setup Toplu AC controls and controller."""
        self._toplu_ac_master_checkbox = self.findChild(QCheckBox, "PriestPartyTopluMasterCheckBox")
        self._toplu_ac_primary_combo = self.findChild(QComboBox, "PriestPartyTopluMasterPrimaryComboBox")
        self._toplu_ac_function_combo = self.findChild(QComboBox, "PriestPartyTopluMasterFunctionComboBox")
        self._toplu_ac_precure_checkbox = self.findChild(QCheckBox, "PriestPartyTopluPreCureMiniCheckBox")

        status_callback = self._make_threadsafe_status_callback()
        try:
            self._toplu_ac_controller = TopluAcMicroController(status_callback=status_callback)
        except Exception as exc:  # pragma: no cover - optional deps
            self._toplu_ac_controller = None
            self._show_status(f"Toplu AC mikro başlatılamadı: {exc}", 6000)
            print(f"Toplu AC controller could not be initialized: {exc}", file=sys.stderr)

        if self._global_start_checkbox is not None:
            self._global_start_checkbox.toggled.connect(lambda _=False: self._sync_toplu_ac_configuration())
        if self._toplu_ac_master_checkbox is not None:
            self._toplu_ac_master_checkbox.toggled.connect(lambda _=False: self._sync_toplu_ac_configuration())
        if self._toplu_ac_primary_combo is not None:
            self._toplu_ac_primary_combo.currentIndexChanged.connect(lambda _=0: self._sync_toplu_ac_configuration())
        if self._toplu_ac_function_combo is not None:
            self._toplu_ac_function_combo.currentIndexChanged.connect(lambda _=0: self._sync_toplu_ac_configuration())
        if self._toplu_ac_precure_checkbox is not None:
            self._toplu_ac_precure_checkbox.toggled.connect(lambda _=False: self._sync_toplu_ac_configuration())

        self._sync_toplu_ac_configuration()

    def _initialize_toplu_cure_controls(self) -> None:
        """Setup Toplu Cure controls and controller."""
        self._toplu_cure_row = self._find_toplu_cure_row()

        status_callback = self._make_threadsafe_status_callback()
        try:
            self._toplu_cure_controller = TopluCureMicroController(status_callback=status_callback)
        except Exception as exc:  # pragma: no cover - optional deps
            self._toplu_cure_controller = None
            self._show_status(f"Toplu Cure mikro başlatılamadı: {exc}", 6000)
            print(f"Toplu Cure controller could not be initialized: {exc}", file=sys.stderr)

        if self._global_start_checkbox is not None:
            self._global_start_checkbox.toggled.connect(lambda _=False: self._sync_toplu_cure_configuration())
        if self._toplu_cure_row is not None:
            self._toplu_cure_row.checkbox.toggled.connect(lambda _=False: self._sync_toplu_cure_configuration())
            self._toplu_cure_row.primary_combo.currentIndexChanged.connect(lambda _=0: self._sync_toplu_cure_configuration())
            self._toplu_cure_row.function_combo.currentIndexChanged.connect(lambda _=0: self._sync_toplu_cure_configuration())

        self._sync_toplu_cure_configuration()

    def _initialize_toplu_buff_controls(self) -> None:
        """Setup Toplu Buff controls and controller."""
        self._toplu_buff_row = self._find_toplu_buff_row()
        self._toplu_buff_precure_checkbox = self.findChild(QCheckBox, "PriestPartyTopluBuffPreCureMiniCheckBox")

        status_callback = self._make_threadsafe_status_callback()
        try:
            self._toplu_buff_controller = TopluBuffMicroController(status_callback=status_callback)
        except Exception as exc:  # pragma: no cover - optional deps
            self._toplu_buff_controller = None
            self._show_status(f"Toplu Buff mikro başlatılamadı: {exc}", 6000)
            print(f"Toplu Buff controller could not be initialized: {exc}", file=sys.stderr)

        if self._global_start_checkbox is not None:
            self._global_start_checkbox.toggled.connect(lambda _=False: self._sync_toplu_buff_configuration())
        if self._toplu_buff_row is not None:
            self._toplu_buff_row.checkbox.toggled.connect(lambda _=False: self._sync_toplu_buff_configuration())
            self._toplu_buff_row.primary_combo.currentIndexChanged.connect(lambda _=0: self._sync_toplu_buff_configuration())
            self._toplu_buff_row.function_combo.currentIndexChanged.connect(lambda _=0: self._sync_toplu_buff_configuration())
        if self._toplu_buff_precure_checkbox is not None:
            self._toplu_buff_precure_checkbox.toggled.connect(lambda _=False: self._sync_toplu_buff_configuration())

        self._sync_toplu_buff_configuration()

    def _initialize_restore_controls(self) -> None:
        """Setup Restore controls and controller."""
        self._restore_master_checkbox = self.findChild(QCheckBox, "PriestPartyRestoreMasterCheckBox")
        self._restore_primary_combo = self.findChild(QComboBox, "PriestPartyRestoreMasterPrimaryComboBox")
        self._restore_function_combo = self.findChild(QComboBox, "PriestPartyRestoreMasterFunctionComboBox")

        status_callback = self._make_threadsafe_status_callback()
        try:
            self._restore_controller = RestoreMicroController(status_callback=status_callback)
        except Exception as exc:  # pragma: no cover - optional deps
            self._restore_controller = None
            self._show_status(f"Restore mikro başlatılamadı: {exc}", 6000)
            print(f"Restore controller could not be initialized: {exc}", file=sys.stderr)

        if self._global_start_checkbox is not None:
            self._global_start_checkbox.toggled.connect(lambda _=False: self._sync_restore_configuration())
        if self._restore_master_checkbox is not None:
            self._restore_master_checkbox.toggled.connect(lambda _=False: self._sync_restore_configuration())
        if self._restore_primary_combo is not None:
            self._restore_primary_combo.currentIndexChanged.connect(lambda _=0: self._sync_restore_configuration())
        if self._restore_function_combo is not None:
            self._restore_function_combo.currentIndexChanged.connect(lambda _=0: self._sync_restore_configuration())
        self._sync_restore_configuration()

    def _initialize_parazit_controls(self) -> None:
        """Setup Parazit controls and controller."""
        self._parazit_row = self._find_parazit_row()
        self._parazit_timer_checkbox = self.findChild(QCheckBox, "PriestAttack3TimerMiniCheckBox")
        self._parazit_partyms_checkbox = self.findChild(QCheckBox, "PriestAttack3PartyMsMiniCheckBox")
        self._parazit_interval_spin = self.findChild(QSpinBox, "PriestAttack3IntervalSpinBox")

        status_callback = self._make_threadsafe_status_callback()
        try:
            self._parazit_controller = ParazitMicroController(status_callback=status_callback)
        except Exception as exc:  # pragma: no cover - optional deps
            self._parazit_controller = None
            self._show_status(f"Parazit mikro başlatılamadı: {exc}", 6000)
            print(f"Parazit controller could not be initialized: {exc}", file=sys.stderr)

        if self._global_start_checkbox is not None:
            self._global_start_checkbox.toggled.connect(lambda _=False: self._sync_parazit_configuration())
        if self._parazit_row is not None:
            self._parazit_row.checkbox.toggled.connect(lambda _=False: self._sync_parazit_configuration())
            self._parazit_row.primary_combo.currentIndexChanged.connect(lambda _=0: self._sync_parazit_configuration())
            self._parazit_row.function_combo.currentIndexChanged.connect(lambda _=0: self._sync_parazit_configuration())

        if self._parazit_timer_checkbox is not None:
            self._parazit_timer_checkbox.toggled.connect(self._on_parazit_timer_toggled)
        if self._parazit_partyms_checkbox is not None:
            self._parazit_partyms_checkbox.toggled.connect(self._on_parazit_partyms_toggled)
        if self._parazit_interval_spin is not None:
            self._parazit_interval_spin.valueChanged.connect(lambda _=0: self._sync_parazit_configuration())

        self._sync_parazit_configuration()

    def _initialize_toplu_controls(self) -> None:
        """Setup Toplu 10k controls and controller."""
        self._toplu_row = self._find_toplu_row()

        status_callback = self._make_threadsafe_status_callback()
        try:
            self._toplu_controller = Toplu10kMicroController(status_callback=status_callback)
        except Exception as exc:  # pragma: no cover - optional deps
            self._toplu_controller = None
            self._show_status(f"Toplu 10k mikro başlatılamadı: {exc}", 6000)
            print(f"Toplu 10k controller could not be initialized: {exc}", file=sys.stderr)

        if self._global_start_checkbox is not None:
            self._global_start_checkbox.toggled.connect(lambda _=False: self._sync_toplu_configuration())
        if self._toplu_row is not None:
            self._toplu_row.checkbox.toggled.connect(lambda _=False: self._sync_toplu_configuration())
            self._toplu_row.primary_combo.currentIndexChanged.connect(lambda _=0: self._sync_toplu_configuration())
            self._toplu_row.function_combo.currentIndexChanged.connect(lambda _=0: self._sync_toplu_configuration())

        self._sync_toplu_configuration()

    def _make_threadsafe_status_callback(self) -> Callable[[str, int], None]:
        """Return a callable that proxies status updates onto the UI thread."""

        def callback(message: str, timeout: int = 4000) -> None:
            QTimer.singleShot(0, lambda msg=message, tout=timeout: self._show_status(msg, tout))

        return callback

    def _populate_shortcut_combo(self, combo: QComboBox) -> None:
        """Populate shortcut combo with Turkish Q keyboard keys."""
        combo.clear()
        combo.addItem("Seç", None)
        for key in AVAILABLE_SHORTCUT_KEYS:
            if key == "space":
                display = "Space"
            elif len(key) == 1:
                display = key.upper()
            else:
                display = key.upper()
            combo.addItem(display, key)

    def _ensure_shortcut_combo(self, combo: Optional[QComboBox]) -> Optional[QComboBox]:
        """Populate shortcut combo if it has not been filled yet."""
        if combo is None:
            return None
        if combo.count() <= 1:
            blocker = QSignalBlocker(combo)
            self._populate_shortcut_combo(combo)
            combo.setCurrentIndex(0)
            del blocker
        return combo

    def _find_cure_row(self) -> Optional[SkillRow]:
        """Locate the Cure row within the priest skill table."""
        for row in self._skill_rows.get("Priest", []):
            if row.label.objectName() == "PriestSupport1Label":
                return row
        return None

    def _find_undy_row(self) -> Optional[SkillRow]:
        """Locate the Undy row within the priest skill table."""
        for row in self._skill_rows.get("Priest", []):
            if row.label.objectName() == "PriestSupport2Label":
                return row
        return None

    def _find_ac_row(self) -> Optional[SkillRow]:
        """Locate the AC row within the priest skill table."""
        for row in self._skill_rows.get("Priest", []):
            if row.label.objectName() == "PriestSupport3Label":
                return row
        return None

    def _find_str30_row(self) -> Optional[SkillRow]:
        """Locate the STR 30 row within the priest skill table."""
        for row in self._skill_rows.get("Priest", []):
            if row.label.objectName() == "PriestSupport4Label":
                return row
        return None

    def _find_zehir_row(self) -> Optional[SkillRow]:
        """Locate the Zehir Cure row within the priest skill table."""
        for row in self._skill_rows.get("Priest", []):
            if row.label.objectName() == "PriestSupport6Label":
                return row
        return None

    def _find_rsiz_row(self) -> Optional[SkillRow]:
        """Locate the R'siz attack row within the priest skill table."""
        for row in self._skill_rows.get("Priest", []):
            if row.label.objectName() == "PriestAttack1Label":
                return row
        return None

    def _find_rr_row(self) -> Optional[SkillRow]:
        """Locate the RR-skill row within the priest attack table."""
        for row in self._skill_rows.get("Priest", []):
            if row.label.objectName() == "PriestAttack2Label":
                return row
        return None

    def _find_malice_row(self) -> Optional[SkillRow]:
        """Locate the Malice row within the priest attack table."""
        for row in self._skill_rows.get("Priest", []):
            if row.label.objectName() == "PriestAttack4Label":
                return row
        return None

    def _find_tekli_row(self) -> Optional[SkillRow]:
        """Locate the Tekli Atak Kırma row within the priest attack table."""
        for row in self._skill_rows.get("Priest", []):
            if row.label.objectName() == "PriestAttack5Label":
                return row
        return None

    def _find_subside_row(self) -> Optional[SkillRow]:
        """Locate the Subside row within the priest attack table."""
        for row in self._skill_rows.get("Priest", []):
            if row.label.objectName() == "PriestAttack6Label":
                return row
        return None

    def _find_torment_row(self) -> Optional[SkillRow]:
        """Locate the Torment row within the priest attack table."""
        for row in self._skill_rows.get("Priest", []):
            if row.label.objectName() == "PriestAttack7Label":
                return row
        return None

    def _find_parazit_row(self) -> Optional[SkillRow]:
        """Locate the Parazit row within the priest attack table."""
        for row in self._skill_rows.get("Priest", []):
            if row.label.objectName() == "PriestAttack3Label":
                return row
        return None

    def _find_rr_row(self) -> Optional[SkillRow]:
        """Locate the RR-skill row within the priest skill table."""
        for row in self._skill_rows.get("Priest", []):
            if row.label.objectName() == "PriestSupport8Label":
                return row
        return None

    def _find_toplu_row(self) -> Optional[SkillRow]:
        """Locate the Toplu 10k row within the priest skill table."""
        for row in self._skill_rows.get("Priest", []):
            if row.label.objectName() == "PriestSupport7Label":
                return row
        return None

    def _find_toplu_buff_row(self) -> Optional[SkillRow]:
        """Locate the Toplu Buff row within the priest party table."""
        for row in self._skill_rows.get("Priest", []):
            if row.label.objectName() == "PriestPartyToplu1Label":
                return row
        return None

    def _find_toplu_cure_row(self) -> Optional[SkillRow]:
        """Locate the Toplu Cure row within the priest party table."""
        for row in self._skill_rows.get("Priest", []):
            if row.label.objectName() == "PriestPartyToplu2Label":
                return row
        return None

    def _find_mana_row(self) -> Optional[SkillRow]:
        """Locate the Mana row within the priest support table."""
        for row in self._skill_rows.get("Priest", []):
            if row.label.objectName() == "PriestSupport8Label":
                return row
        return None

    def _find_heal_row(self) -> Optional[SkillRow]:
        """Locate the Heal row within the priest support table."""
        for row in self._skill_rows.get("Priest", []):
            if row.label.objectName() == "PriestSupport9Label":
                return row
        return None

    def _on_cure_shortcut_toggled(self, checked: bool) -> None:
        """Enable combo box and resync config when shortcut toggle changes."""
        if self._cure_shortcut_combo is not None:
            self._cure_shortcut_combo.setEnabled(checked)
        if not checked:
            self._release_shortcut("cure", preserve_selection=True)
        self._sync_cure_configuration()

    def _on_cure_shortcut_key_changed(self, index: int) -> None:
        """Handle shortcut key selection with conflict checks."""
        if self._cure_shortcut_combo is None:
            return
        data = self._cure_shortcut_combo.itemData(index)
        if data is None:
            self._release_shortcut("cure", preserve_selection=False)
            self._sync_cure_configuration()
            return
        key = str(data)
        existing = self._priest_shortcut_assignments.get(key)
        if existing and existing != "cure":
            display = self._cure_shortcut_combo.itemText(index)
            self._show_status(f"{display} tuşu Priest içinde başka bir kod tarafından kullanılıyor.", 5000)
            self._restore_shortcut_selection("cure", self._cure_shortcut_combo)
            self._sync_cure_configuration()
            return
        self._assign_shortcut(key, "cure")
        self._sync_cure_configuration()

    def _on_zehir_shortcut_key_changed(self, index: int) -> None:
        """Handle Zehir Cure shortcut selection with conflict checks."""
        if self._zehir_shortcut_combo is None:
            return
        data = self._zehir_shortcut_combo.itemData(index)
        if data is None:
            self._release_shortcut("zehir_cure", preserve_selection=False)
            self._sync_zehir_configuration()
            return
        key = str(data)
        existing = self._priest_shortcut_assignments.get(key)
        if existing and existing != "zehir_cure":
            display = self._zehir_shortcut_combo.itemText(index)
            self._show_status(f"{display} tuşu Priest içinde başka bir kod tarafından kullanılıyor.", 5000)
            self._restore_shortcut_selection("zehir_cure", self._zehir_shortcut_combo)
            self._sync_zehir_configuration()
            return
        self._assign_shortcut(key, "zehir_cure")
        self._sync_zehir_configuration()

    def _on_rsiz_shortcut_changed(self, index: int) -> None:
        """Handle R'siz Atak shortcut selection with conflict checks."""
        if self._rsiz_shortcut_combo is None:
            return
        data = self._rsiz_shortcut_combo.itemData(index)
        if data is None:
            self._release_shortcut("rsiz_atak", preserve_selection=False)
            self._sync_rsiz_configuration()
            return
        key = str(data)
        existing = self._priest_shortcut_assignments.get(key)
        if existing and existing != "rsiz_atak":
            display = self._rsiz_shortcut_combo.itemText(index)
            self._show_status(f"{display} tuşu Priest içinde başka bir kod tarafından kullanılıyor.", 5000)
            self._restore_shortcut_selection("rsiz_atak", self._rsiz_shortcut_combo)
            self._sync_rsiz_configuration()
            return
        self._assign_shortcut(key, "rsiz_atak")
        self._sync_rsiz_configuration()

    def _on_rr_shortcut_changed(self, index: int) -> None:
        """Handle RR skill shortcut selection with conflict checks."""
        if self._rr_shortcut_combo is None:
            return
        data = self._rr_shortcut_combo.itemData(index)
        if data is None:
            self._release_shortcut("rr_skill", preserve_selection=False)
            self._sync_rr_configuration()
            return
        key = str(data)
        existing = self._priest_shortcut_assignments.get(key)
        if existing and existing != "rr_skill":
            display = self._rr_shortcut_combo.itemText(index)
            self._show_status(f"{display} tuşu Priest içinde başka bir kod tarafından kullanılıyor.", 5000)
            self._restore_shortcut_selection("rr_skill", self._rr_shortcut_combo)
            self._sync_rr_configuration()
            return
        self._assign_shortcut(key, "rr_skill")
        self._sync_rr_configuration()

    def _on_parazit_timer_toggled(self, checked: bool) -> None:
        """Ensure only one Parazit mode is active and sync configuration."""
        if checked and self._parazit_partyms_checkbox is not None:
            self._parazit_partyms_checkbox.setChecked(False)
        self._sync_parazit_configuration()

    def _on_parazit_partyms_toggled(self, checked: bool) -> None:
        """Ensure only one Parazit mode is active and sync configuration."""
        if checked and self._parazit_timer_checkbox is not None:
            self._parazit_timer_checkbox.setChecked(False)
        self._sync_parazit_configuration()

    def _on_malice_timer_toggled(self, checked: bool) -> None:
        """Ensure only one Malice mode is active and sync configuration."""
        if checked and self._malice_partyms_checkbox is not None:
            self._malice_partyms_checkbox.setChecked(False)
        self._sync_malice_configuration()

    def _on_malice_partyms_toggled(self, checked: bool) -> None:
        """Ensure only one Malice mode is active and sync configuration."""
        if checked and self._malice_timer_checkbox is not None:
            self._malice_timer_checkbox.setChecked(False)
        self._sync_malice_configuration()

    def _on_tekli_timer_toggled(self, checked: bool) -> None:
        """Ensure only one Tekli mode is active and sync configuration."""
        if checked and self._tekli_partyms_checkbox is not None:
            self._tekli_partyms_checkbox.setChecked(False)
        self._sync_tekli_configuration()

    def _on_tekli_partyms_toggled(self, checked: bool) -> None:
        """Ensure only one Tekli mode is active and sync configuration."""
        if checked and self._tekli_timer_checkbox is not None:
            self._tekli_timer_checkbox.setChecked(False)
        self._sync_tekli_configuration()

    def _on_shared_party_hp_changed(self) -> None:
        """Propagate shared party HP control changes to dependent configs."""
        self._sync_restore_configuration()
        self._sync_toplu_configuration()

    def _on_party_shared_calibrate_clicked(self) -> None:
        """Launch shared HP calibration once and refresh dependent controllers."""
        if self._party_shared_calibration_running:
            self._show_status("Kalibrasyon zaten devam ediyor.", 4000)
            return

        slots = DEFAULT_PARTY_SIZE
        if self._party_shared_party_size_spin is not None:
            try:
                slots = int(self._party_shared_party_size_spin.value())
            except Exception:
                slots = DEFAULT_PARTY_SIZE
        slots = max(2, min(slots, 8))

        controllers = [self._restore_controller, self._toplu_controller]
        primary_controller = next((controller for controller in controllers if controller is not None), None)
        if primary_controller is None:
            self._show_status("Restore veya Toplu 10k mikro bulunamadı; kalibrasyon başlatılamadı.", 5000)
            return

        status_callback = self._make_threadsafe_status_callback()

        def run_calibration() -> None:
            try:
                status_callback("Lütfen party’deki X butonunun tam ortasına tıklayın.", 6000)
                calibrate = getattr(primary_controller, "calibrate", None)
                if callable(calibrate):
                    calibrate(slots)
                for controller in controllers:
                    if controller is None or controller is primary_controller:
                        continue
                    invalidate = getattr(controller, "invalidate_calibration_cache", None)
                    if callable(invalidate):
                        invalidate()
                status_callback("Party HP kalibrasyonu tamamlandı.", 5000)
            except Exception as exc:  # pragma: no cover - kullanıcı etkileşimi
                status_callback(f"Kalibrasyon başarısız: {exc}", 6000)
            finally:
                self._party_shared_calibration_running = False
                QTimer.singleShot(0, self._on_shared_party_hp_changed)

        self._party_shared_calibration_running = True
        threading.Thread(target=run_calibration, name="PartyHpCalibrate", daemon=True).start()

    def _on_rr_shortcut_changed(self, index: int) -> None:
        """Handle RR-skill shortcut selection with conflict checks."""
        if self._rr_shortcut_combo is None:
            return
        data = self._rr_shortcut_combo.itemData(index)
        if data is None:
            self._release_shortcut("rr_skill", preserve_selection=False)
            self._sync_rr_configuration()
            return
        key = str(data)
        existing = self._priest_shortcut_assignments.get(key)
        if existing and existing != "rr_skill":
            display = self._rr_shortcut_combo.itemText(index)
            self._show_status(f"{display} tuşu Priest içinde başka bir kod tarafından kullanılıyor.", 5000)
            self._restore_shortcut_selection("rr_skill", self._rr_shortcut_combo)
            self._sync_rr_configuration()
            return
        self._assign_shortcut(key, "rr_skill")
        self._sync_rr_configuration()

    def _get_selected_shortcut(self, identifier: str) -> Optional[str]:
        """Return stored shortcut key for identifier."""
        return self._shortcut_selected_keys.get(identifier)

    def _set_selected_shortcut(self, identifier: str, key: Optional[str]) -> None:
        """Store or clear shortcut key for identifier."""
        if key is None:
            self._shortcut_selected_keys.pop(identifier, None)
        else:
            self._shortcut_selected_keys[identifier] = key

    def _restore_shortcut_selection(self, identifier: str, combo: Optional[QComboBox]) -> None:
        """Restore combo selection based on stored shortcut."""
        if combo is None:
            return
        blocker = QSignalBlocker(combo)
        target = self._get_selected_shortcut(identifier)
        if target is None:
            combo.setCurrentIndex(0)
        else:
            target_index = combo.findData(target)
            if target_index == -1:
                combo.setCurrentIndex(0)
                self._set_selected_shortcut(identifier, None)
            else:
                combo.setCurrentIndex(target_index)
        del blocker

    def _assign_shortcut(self, key: str, identifier: str) -> None:
        """Assign shortcut key to identifier, ensuring previous key cleared."""
        self._release_shortcut(identifier, preserve_selection=True)
        self._priest_shortcut_assignments[key] = identifier
        self._set_selected_shortcut(identifier, key)

    def _release_shortcut(self, identifier: str, *, preserve_selection: bool) -> None:
        """Remove shortcut assignment for identifier."""
        for assigned_key, owner in list(self._priest_shortcut_assignments.items()):
            if owner == identifier:
                del self._priest_shortcut_assignments[assigned_key]
        if not preserve_selection:
            self._set_selected_shortcut(identifier, None)

    def _extract_primary_key(self, combo: Optional[QComboBox]) -> Optional[str]:
        """Read primary slot combo selection."""
        if combo is None:
            return None
        text = combo.currentText().strip()
        if not text or text.lower() == "seç":
            return None
        return text.lower()

    def _extract_function_key(self, combo: Optional[QComboBox]) -> Optional[str]:
        """Read function row combo selection."""
        if combo is None:
            return None
        text = combo.currentText().strip()
        if not text or text.lower() == "seç":
            return None
        return text.lower()

    def _build_cure_config(self) -> CureConfig:
        """Gather Cure configuration from UI elements."""
        config = CureConfig()
        if self._global_start_checkbox is not None:
            config.global_enabled = self._global_start_checkbox.isChecked()
        if self._cure_row is not None:
            config.skill_enabled = self._cure_row.checkbox.isChecked()
            config.primary_key = self._extract_primary_key(self._cure_row.primary_combo)
            config.function_key = self._extract_function_key(self._cure_row.function_combo)
        if self._cure_auto_checkbox is not None:
            config.auto_enabled = self._cure_auto_checkbox.isChecked()
        if self._cure_shortcut_checkbox is not None:
            config.shortcut_enabled = self._cure_shortcut_checkbox.isChecked()
        if config.shortcut_enabled and self._cure_shortcut_combo is not None:
            data = self._cure_shortcut_combo.currentData()
            if data:
                config.shortcut_key = str(data)
            else:
                config.shortcut_enabled = False
                config.shortcut_key = None
        return config

    def _sync_cure_configuration(self) -> None:
        """Synchronize UI state with Cure controller."""
        if self._cure_controller is None:
            return
        config = self._build_cure_config()
        shortcut_key = config.shortcut_key
        if not config.skill_enabled or not config.shortcut_enabled:
            self._release_shortcut("cure", preserve_selection=True)
        elif shortcut_key:
            existing = self._priest_shortcut_assignments.get(shortcut_key)
            if existing and existing != "cure":
                display = shortcut_key.upper()
                if self._cure_shortcut_combo is not None:
                    display = self._cure_shortcut_combo.currentText()
                self._show_status(f"{display} tuşu Priest içinde başka kod için kullanılıyor.", 5000)
                self._restore_shortcut_selection("cure", self._cure_shortcut_combo)
                config.shortcut_enabled = False
                config.shortcut_key = None
            else:
                self._assign_shortcut(shortcut_key, "cure")
        self._cure_controller.update_config(config)

    def _build_undy_config(self) -> UndyConfig:
        """Gather Undy configuration from UI elements."""
        config = UndyConfig()
        if self._global_start_checkbox is not None:
            config.global_enabled = self._global_start_checkbox.isChecked()
        if self._undy_row is not None:
            config.skill_enabled = self._undy_row.checkbox.isChecked()
            config.primary_key = self._extract_primary_key(self._undy_row.primary_combo)
            config.function_key = self._extract_function_key(self._undy_row.function_combo)
        return config

    def _sync_undy_configuration(self) -> None:
        """Synchronize UI state with Undy controller."""
        if self._undy_controller is None:
            return
        config = self._build_undy_config()
        self._undy_controller.update_config(config)

    def _build_ac_config(self) -> AcConfig:
        """Gather AC configuration from UI elements."""
        config = AcConfig()
        if self._global_start_checkbox is not None:
            config.global_enabled = self._global_start_checkbox.isChecked()
        if self._ac_row is not None:
            config.skill_enabled = self._ac_row.checkbox.isChecked()
            config.primary_key = self._extract_primary_key(self._ac_row.primary_combo)
            config.function_key = self._extract_function_key(self._ac_row.function_combo)
        return config

    def _sync_ac_configuration(self) -> None:
        """Synchronize UI state with AC controller."""
        if self._ac_controller is None:
            return
        config = self._build_ac_config()
        self._ac_controller.update_config(config)

    def _build_str30_config(self) -> Str30Config:
        """Gather STR 30 configuration from UI elements."""
        config = Str30Config()
        if self._global_start_checkbox is not None:
            config.global_enabled = self._global_start_checkbox.isChecked()
        if self._str30_row is not None:
            config.skill_enabled = self._str30_row.checkbox.isChecked()
            config.primary_key = self._extract_primary_key(self._str30_row.primary_combo)
            config.function_key = self._extract_function_key(self._str30_row.function_combo)
        return config

    def _sync_str30_configuration(self) -> None:
        """Synchronize UI state with STR 30 controller."""
        if self._str30_controller is None:
            return
        config = self._build_str30_config()
        self._str30_controller.update_config(config)

    def _build_zehir_config(self) -> PoisonCureConfig:
        """Gather Zehir Cure configuration from UI elements."""
        config = PoisonCureConfig()
        if self._global_start_checkbox is not None:
            config.global_enabled = self._global_start_checkbox.isChecked()
        if self._zehir_row is not None:
            config.skill_enabled = self._zehir_row.checkbox.isChecked()
            config.primary_key = self._extract_primary_key(self._zehir_row.primary_combo)
            config.function_key = self._extract_function_key(self._zehir_row.function_combo)
        if self._zehir_shortcut_combo is not None:
            data = self._zehir_shortcut_combo.currentData()
            if data:
                config.shortcut_key = str(data)
        return config

    def _sync_zehir_configuration(self) -> None:
        """Synchronize UI state with Zehir Cure controller."""
        if self._zehir_controller is None:
            return
        config = self._build_zehir_config()
        if not config.skill_enabled:
            self._release_shortcut("zehir_cure", preserve_selection=True)
            self._zehir_controller.update_config(config)
            return
        elif not config.shortcut_key:
            self._release_shortcut("zehir_cure", preserve_selection=False)
            self._zehir_controller.update_config(config)
            return
        existing = self._priest_shortcut_assignments.get(config.shortcut_key)
        if existing and existing != "zehir_cure":
            display = config.shortcut_key.upper()
            if self._zehir_shortcut_combo is not None:
                display = self._zehir_shortcut_combo.currentText()
            self._show_status(f"{display} tuşu Priest içinde başka bir kod tarafından kullanılıyor.", 5000)
            self._restore_shortcut_selection("zehir_cure", self._zehir_shortcut_combo)
            self._release_shortcut("zehir_cure", preserve_selection=False)
            config.shortcut_key = None
            self._zehir_controller.update_config(config)
            return
        self._assign_shortcut(config.shortcut_key, "zehir_cure")
        self._zehir_controller.update_config(config)

    def _build_rsiz_config(self) -> RsizAtakConfig:
        """Gather R'siz Atak configuration from UI elements."""
        config = RsizAtakConfig()
        if self._global_start_checkbox is not None:
            config.global_enabled = self._global_start_checkbox.isChecked()
        if self._rsiz_row is not None:
            config.skill_enabled = self._rsiz_row.checkbox.isChecked()
            config.primary_key = self._extract_primary_key(self._rsiz_row.primary_combo)
        if self._rsiz_shortcut_combo is not None:
            data = self._rsiz_shortcut_combo.currentData()
            if data:
                config.shortcut_key = str(data)
        return config

    def _sync_rsiz_configuration(self) -> None:
        """Synchronize UI state with R'siz Atak controller."""
        if self._rsiz_controller is None:
            return
        config = self._build_rsiz_config()
        if not config.skill_enabled:
            self._release_shortcut("rsiz_atak", preserve_selection=True)
            self._rsiz_controller.update_config(config)
            return
        if not config.shortcut_key:
            self._release_shortcut("rsiz_atak", preserve_selection=False)
            self._rsiz_controller.update_config(config)
            return
        existing = self._priest_shortcut_assignments.get(config.shortcut_key)
        if existing and existing != "rsiz_atak":
            display = config.shortcut_key.upper()
            if self._rsiz_shortcut_combo is not None:
                display = self._rsiz_shortcut_combo.currentText()
            self._show_status(f"{display} tuşu Priest içinde başka bir kod tarafından kullanılıyor.", 5000)
            self._restore_shortcut_selection("rsiz_atak", self._rsiz_shortcut_combo)
            self._release_shortcut("rsiz_atak", preserve_selection=False)
            config.shortcut_key = None
            self._rsiz_controller.update_config(config)
            return
        self._assign_shortcut(config.shortcut_key, "rsiz_atak")
        self._rsiz_controller.update_config(config)

    def _build_rr_config(self) -> RrSkillConfig:
        """Gather RR skill configuration from UI elements."""
        config = RrSkillConfig()
        if self._global_start_checkbox is not None:
            config.global_enabled = self._global_start_checkbox.isChecked()
        if self._rr_row is not None:
            config.skill_enabled = self._rr_row.checkbox.isChecked()
            config.primary_key = self._extract_primary_key(self._rr_row.primary_combo)
        if self._rr_shortcut_combo is not None:
            data = self._rr_shortcut_combo.currentData()
            if data:
                config.shortcut_key = str(data)
        return config

    def _sync_rr_configuration(self) -> None:
        """Synchronize UI state with RR skill controller."""
        if self._rr_controller is None:
            return
        config = self._build_rr_config()
        if not config.skill_enabled:
            self._release_shortcut("rr_skill", preserve_selection=True)
            self._rr_controller.update_config(config)
            return
        if not config.shortcut_key:
            self._release_shortcut("rr_skill", preserve_selection=False)
            self._rr_controller.update_config(config)
            return
        existing = self._priest_shortcut_assignments.get(config.shortcut_key)
        if existing and existing != "rr_skill":
            display = config.shortcut_key.upper()
            if self._rr_shortcut_combo is not None:
                display = self._rr_shortcut_combo.currentText()
            self._show_status(f"{display} tuşu Priest içinde başka bir kod tarafından kullanılıyor.", 5000)
            self._restore_shortcut_selection("rr_skill", self._rr_shortcut_combo)
            self._release_shortcut("rr_skill", preserve_selection=False)
            config.shortcut_key = None
            self._rr_controller.update_config(config)
            return
        self._assign_shortcut(config.shortcut_key, "rr_skill")
        self._rr_controller.update_config(config)

    def _build_parazit_config(self) -> ParazitConfig:
        """Gather Parazit configuration from UI elements."""
        config = ParazitConfig()
        if self._global_start_checkbox is not None:
            config.global_enabled = self._global_start_checkbox.isChecked()
        if self._parazit_row is not None:
            config.skill_enabled = self._parazit_row.checkbox.isChecked()
            config.primary_key = self._extract_primary_key(self._parazit_row.primary_combo)
            config.function_key = self._extract_function_key(self._parazit_row.function_combo)
        if self._parazit_timer_checkbox is not None and self._parazit_timer_checkbox.isChecked():
            config.mode = "timer"
        elif self._parazit_partyms_checkbox is not None and self._parazit_partyms_checkbox.isChecked():
            config.mode = "partyms"
        if self._parazit_interval_spin is not None:
            config.interval_seconds = float(self._parazit_interval_spin.value())
        return config

    def _sync_parazit_configuration(self) -> None:
        """Synchronize UI state with Parazit controller."""
        if self._parazit_controller is None:
            return
        config = self._build_parazit_config()
        self._parazit_controller.update_config(config)
        if self._malice_controller is not None:
            self._sync_malice_configuration()

    def _build_malice_config(self) -> MaliceConfig:
        """Gather Malice configuration from UI elements."""
        config = MaliceConfig()
        if self._global_start_checkbox is not None:
            config.global_enabled = self._global_start_checkbox.isChecked()
        if self._malice_row is not None:
            config.skill_enabled = self._malice_row.checkbox.isChecked()
            config.primary_key = self._extract_primary_key(self._malice_row.primary_combo)
            config.function_key = self._extract_function_key(self._malice_row.function_combo)
        if self._malice_timer_checkbox is not None and self._malice_timer_checkbox.isChecked():
            config.mode = "timer"
        elif self._malice_partyms_checkbox is not None and self._malice_partyms_checkbox.isChecked():
            config.mode = "parazit"
        if self._malice_interval_spin is not None:
            try:
                config.interval_seconds = float(self._malice_interval_spin.value())
            except Exception:
                config.interval_seconds = 10.0
        return config

    def _unregister_malice_from_parazit(self) -> None:
        if self._malice_registered_with_parazit and self._parazit_controller is not None and self._malice_controller is not None:
            self._parazit_controller.unregister_completion_callback(self._malice_controller.handle_parazit_completion)
        self._malice_registered_with_parazit = False

    def _sync_malice_configuration(self) -> None:
        """Synchronize UI state with Malice controller."""
        if self._malice_controller is None:
            return
        config = self._build_malice_config()
        if not config.skill_enabled:
            self._unregister_malice_from_parazit()
            self._malice_controller.update_config(config)
            return
        if not config.mode:
            self._show_status("Malice için 'Zamanlı' veya 'Parazit'ten Sonra' modlarından birini seçin.", 4000)
            self._unregister_malice_from_parazit()
            self._malice_controller.update_config(config)
            return
        if config.mode == "parazit" and config.parazit_ready():
            if self._parazit_controller is None:
                self._show_status("Parazit mikro etkin değil; Malice tetiklenemedi.", 5000)
                self._unregister_malice_from_parazit()
            else:
                if not self._malice_registered_with_parazit:
                    self._parazit_controller.register_completion_callback(self._malice_controller.handle_parazit_completion)
                    self._malice_registered_with_parazit = True
        else:
            self._unregister_malice_from_parazit()
        self._malice_controller.update_config(config)
        self._sync_tekli_configuration()

    def _unregister_tekli_from_malice(self) -> None:
        if self._tekli_registered_with_malice and self._malice_controller is not None and self._tekli_controller is not None:
            self._malice_controller.unregister_completion_callback(self._tekli_controller.handle_malice_completion)
        self._tekli_registered_with_malice = False

    def _build_tekli_config(self) -> TekliAtakKirmaConfig:
        """Gather Tekli Atak Kırma configuration from UI elements."""
        config = TekliAtakKirmaConfig()
        if self._global_start_checkbox is not None:
            config.global_enabled = self._global_start_checkbox.isChecked()
        if self._tekli_row is not None:
            config.skill_enabled = self._tekli_row.checkbox.isChecked()
            config.primary_key = self._extract_primary_key(self._tekli_row.primary_combo)
            config.function_key = self._extract_function_key(self._tekli_row.function_combo)
        if self._tekli_timer_checkbox is not None and self._tekli_timer_checkbox.isChecked():
            config.mode = "timer"
        elif self._tekli_partyms_checkbox is not None and self._tekli_partyms_checkbox.isChecked():
            config.mode = "malice"
        if self._tekli_interval_spin is not None:
            try:
                config.interval_seconds = float(self._tekli_interval_spin.value())
            except Exception:
                config.interval_seconds = 10.0
        return config

    def _sync_tekli_configuration(self) -> None:
        """Synchronize UI state with Tekli Atak Kırma controller."""
        if self._tekli_controller is None:
            return
        config = self._build_tekli_config()
        if not config.skill_enabled:
            self._unregister_tekli_from_malice()
            self._tekli_controller.update_config(config)
            return
        if not config.mode:
            self._show_status("Tekli Atak Kırma için 'Zamanlı' veya 'Malice'ten Sonra' modlarından birini seçin.", 4000)
            self._unregister_tekli_from_malice()
            self._tekli_controller.update_config(config)
            return
        if config.mode == "malice":
            if self._malice_controller is None:
                self._show_status("Malice mikro etkin değil; Tekli Atak Kırma tetiklenemedi.", 5000)
                self._unregister_tekli_from_malice()
            else:
                if not self._tekli_registered_with_malice:
                    self._malice_controller.register_completion_callback(self._tekli_controller.handle_malice_completion)
                    self._tekli_registered_with_malice = True
        else:
            self._unregister_tekli_from_malice()
        self._tekli_controller.update_config(config)

    def _build_subside_config(self) -> SubsideConfig:
        """Gather Subside configuration from UI elements."""
        config = SubsideConfig()
        if self._global_start_checkbox is not None:
            config.global_enabled = self._global_start_checkbox.isChecked()
        if self._subside_row is not None:
            config.skill_enabled = self._subside_row.checkbox.isChecked()
            config.primary_key = self._extract_primary_key(self._subside_row.primary_combo)
            config.function_key = self._extract_function_key(self._subside_row.function_combo)
        if self._subside_interval_spin is not None:
            try:
                config.interval_seconds = float(self._subside_interval_spin.value())
            except Exception:
                config.interval_seconds = 15.0
        return config

    def _sync_subside_configuration(self) -> None:
        """Synchronize UI state with Subside controller."""
        if self._subside_controller is None:
            return
        config = self._build_subside_config()
        self._subside_controller.update_config(config)

    def _build_torment_config(self) -> TormentConfig:
        """Gather Torment configuration from UI elements."""
        config = TormentConfig()
        if self._global_start_checkbox is not None:
            config.global_enabled = self._global_start_checkbox.isChecked()
        if self._torment_row is not None:
            config.skill_enabled = self._torment_row.checkbox.isChecked()
            config.primary_key = self._extract_primary_key(self._torment_row.primary_combo)
            config.function_key = self._extract_function_key(self._torment_row.function_combo)
        if self._torment_interval_spin is not None:
            try:
                config.interval_seconds = float(self._torment_interval_spin.value())
            except Exception:
                config.interval_seconds = 15.0
        return config

    def _sync_torment_configuration(self) -> None:
        """Synchronize UI state with Torment controller."""
        if self._torment_controller is None:
            return
        config = self._build_torment_config()
        self._torment_controller.update_config(config)

    def _build_restore_config(self) -> RestoreConfig:
        """Gather Restore configuration from UI elements."""
        config = RestoreConfig()
        if self._global_start_checkbox is not None:
            config.global_enabled = self._global_start_checkbox.isChecked()
        if self._restore_master_checkbox is not None:
            config.skill_enabled = self._restore_master_checkbox.isChecked()
        size = DEFAULT_PARTY_SIZE
        if self._party_shared_party_size_spin is not None:
            try:
                size = int(self._party_shared_party_size_spin.value())
            except Exception:
                size = DEFAULT_PARTY_SIZE
        config.party_size = max(2, min(size, 8))
        config.primary_key = self._extract_primary_key(self._restore_primary_combo)
        config.function_key = self._extract_function_key(self._restore_function_combo)
        config.threshold = None
        if self._party_shared_threshold_combo is not None:
            text = self._party_shared_threshold_combo.currentText().strip()
            if text.startswith("%"):
                text = text[1:]
            if text.isdigit():
                value = int(text)
                if value in HP_THRESHOLD_VALUES:
                    config.threshold = value
        return config

    def _build_toplu_ac_config(self) -> TopluAcConfig:
        """Gather Toplu AC configuration from UI elements."""
        config = TopluAcConfig()
        if self._global_start_checkbox is not None:
            config.global_enabled = self._global_start_checkbox.isChecked()
        if self._toplu_ac_master_checkbox is not None:
            config.skill_enabled = self._toplu_ac_master_checkbox.isChecked()
        config.primary_key = self._extract_primary_key(self._toplu_ac_primary_combo)
        config.function_key = self._extract_function_key(self._toplu_ac_function_combo)
        if self._toplu_ac_precure_checkbox is not None:
            config.precure_enabled = self._toplu_ac_precure_checkbox.isChecked()
        return config

    def _sync_restore_configuration(self) -> None:
        """Synchronize UI state with Restore controller."""
        if self._restore_controller is None:
            return
        config = self._build_restore_config()
        if config.skill_enabled and config.threshold is None:
            self._show_status("Restore için eşik değerini seçin.", 4000)
        self._restore_controller.update_config(config)

    def _sync_toplu_ac_configuration(self) -> None:
        """Synchronize UI state with Toplu AC controller."""
        if self._toplu_ac_controller is None:
            return
        config = self._build_toplu_ac_config()
        self._toplu_ac_controller.update_config(config)

    def _build_rr_config(self) -> RrSkillConfig:
        """Gather RR-skill configuration from UI elements."""
        config = RrSkillConfig()
        if self._global_start_checkbox is not None:
            config.global_enabled = self._global_start_checkbox.isChecked()
        if self._rr_row is not None:
            config.skill_enabled = self._rr_row.checkbox.isChecked()
            config.primary_key = self._extract_primary_key(self._rr_row.primary_combo)
        if self._rr_shortcut_combo is not None:
            data = self._rr_shortcut_combo.currentData()
            if data:
                config.shortcut_key = str(data)
        return config

    def _sync_rr_configuration(self) -> None:
        """Synchronize UI state with RR-skill controller."""
        if self._rr_controller is None:
            return
        config = self._build_rr_config()
        if not config.skill_enabled:
            self._release_shortcut("rr_skill", preserve_selection=True)
            self._rr_controller.update_config(config)
            return
        if not config.shortcut_key:
            self._release_shortcut("rr_skill", preserve_selection=False)
            self._rr_controller.update_config(config)
            return
        existing = self._priest_shortcut_assignments.get(config.shortcut_key)
        if existing and existing != "rr_skill":
            display = config.shortcut_key.upper()
            if self._rr_shortcut_combo is not None:
                display = self._rr_shortcut_combo.currentText()
            self._show_status(f"{display} tuşu Priest içinde başka bir kod tarafından kullanılıyor.", 5000)
            self._restore_shortcut_selection("rr_skill", self._rr_shortcut_combo)
            self._release_shortcut("rr_skill", preserve_selection=False)
            config.shortcut_key = None
            self._rr_controller.update_config(config)
            return
        self._assign_shortcut(config.shortcut_key, "rr_skill")
        self._rr_controller.update_config(config)

    def _build_mana_config(self) -> ManaConfig:
        """Gather Mana configuration from UI elements."""
        config = ManaConfig()
        if self._global_start_checkbox is not None:
            config.global_enabled = self._global_start_checkbox.isChecked()
        if self._mana_row is not None:
            config.skill_enabled = self._mana_row.checkbox.isChecked()
            config.primary_key = self._extract_primary_key(self._mana_row.primary_combo)
            config.function_key = self._extract_function_key(self._mana_row.function_combo)
        config.threshold = None
        if self._mana_threshold_combo is not None:
            text = self._mana_threshold_combo.currentText().strip()
            if text.startswith("%"):
                text = text[1:]
            if text.isdigit():
                value = int(text)
                if value in HP_THRESHOLD_VALUES:
                    config.threshold = value
        return config

    def _sync_mana_configuration(self) -> None:
        """Synchronize UI state with Mana controller."""
        if self._mana_controller is None:
            return
        config = self._build_mana_config()
        if config.skill_enabled and config.threshold is None:
            self._show_status("Mana için eşik değerini seçin.", 4000)
        self._mana_controller.update_config(config)

    def _build_heal_config(self) -> HealConfig:
        """Gather Heal configuration from UI elements."""
        config = HealConfig()
        if self._global_start_checkbox is not None:
            config.global_enabled = self._global_start_checkbox.isChecked()
        if self._heal_row is not None:
            config.skill_enabled = self._heal_row.checkbox.isChecked()
            config.primary_key = self._extract_primary_key(self._heal_row.primary_combo)
            config.function_key = self._extract_function_key(self._heal_row.function_combo)
        config.threshold = None
        if self._heal_threshold_combo is not None:
            text = self._heal_threshold_combo.currentText().strip()
            if text.startswith("%"):
                text = text[1:]
            if text.isdigit():
                value = int(text)
                if value in HP_THRESHOLD_VALUES:
                    config.threshold = value
        return config

    def _sync_heal_configuration(self) -> None:
        """Synchronize UI state with Heal controller."""
        if self._heal_controller is None:
            return
        config = self._build_heal_config()
        if config.skill_enabled and config.threshold is None:
            self._show_status("Heal için eşik değerini seçin.", 4000)
        self._maybe_show_heal_hint(config)
        self._heal_controller.update_config(config)

    def _maybe_show_heal_hint(self, config: HealConfig) -> None:
        """Display instructional message when Heal is enabled."""
        if not config.skill_enabled:
            self._heal_last_hint = None
            return
        primary = config.primary_key
        if not primary or not primary.isdigit():
            if self._heal_last_hint != "invalid":
                self._show_status("Heal için 0-9 arasında bir numara seçin.", 4000)
            self._heal_last_hint = "invalid"
            return
        if primary == self._heal_last_hint:
            return
        secondary = str((int(primary) + 1) % 10)
        self._show_status(f"Heal mikro {primary} ve {secondary} tuşlarını sırasıyla basar.", 5000)
        self._heal_last_hint = primary

    def _build_toplu_config(self) -> Toplu10kConfig:
        """Gather Toplu 10k configuration from UI elements."""
        config = Toplu10kConfig()
        if self._global_start_checkbox is not None:
            config.global_enabled = self._global_start_checkbox.isChecked()
        if self._toplu_row is not None:
            config.skill_enabled = self._toplu_row.checkbox.isChecked()
            config.primary_key = self._extract_primary_key(self._toplu_row.primary_combo)
            config.function_key = self._extract_function_key(self._toplu_row.function_combo)
        size = DEFAULT_PARTY_SIZE
        if self._party_shared_party_size_spin is not None:
            try:
                size = int(self._party_shared_party_size_spin.value())
            except Exception:
                size = DEFAULT_PARTY_SIZE
        config.party_size = max(2, min(size, 8))
        config.threshold = None
        if self._party_shared_threshold_combo is not None:
            data = self._party_shared_threshold_combo.currentText().strip()
            if data.startswith("%"):
                data = data[1:]
            if data.isdigit():
                value = int(data)
                if value in HP_THRESHOLD_VALUES:
                    config.threshold = value
        return config

    def _sync_toplu_configuration(self) -> None:
        """Synchronize UI state with Toplu 10k controller."""
        if self._toplu_controller is None:
            return
        config = self._build_toplu_config()
        if config.skill_enabled and config.threshold is None:
            self._show_status("Toplu 10k için eşik değerini seçin.", 4000)
        self._toplu_controller.update_config(config)

    def _build_toplu_buff_config(self) -> TopluBuffConfig:
        """Gather Toplu Buff configuration from UI elements."""
        config = TopluBuffConfig()
        if self._global_start_checkbox is not None:
            config.global_enabled = self._global_start_checkbox.isChecked()
        if self._toplu_buff_row is not None:
            config.skill_enabled = self._toplu_buff_row.checkbox.isChecked()
            config.primary_key = self._extract_primary_key(self._toplu_buff_row.primary_combo)
            config.function_key = self._extract_function_key(self._toplu_buff_row.function_combo)
        if self._toplu_buff_precure_checkbox is not None:
            config.precure_enabled = self._toplu_buff_precure_checkbox.isChecked()
        return config

    def _sync_toplu_buff_configuration(self) -> None:
        """Synchronize UI state with Toplu Buff controller."""
        if self._toplu_buff_controller is None:
            return
        config = self._build_toplu_buff_config()
        self._toplu_buff_controller.update_config(config)

    def _build_toplu_cure_config(self) -> TopluCureConfig:
        """Gather Toplu Cure configuration from UI elements."""
        config = TopluCureConfig()
        if self._global_start_checkbox is not None:
            config.global_enabled = self._global_start_checkbox.isChecked()
        if self._toplu_cure_row is not None:
            config.skill_enabled = self._toplu_cure_row.checkbox.isChecked()
            config.primary_key = self._extract_primary_key(self._toplu_cure_row.primary_combo)
            config.function_key = self._extract_function_key(self._toplu_cure_row.function_combo)
        return config

    def _sync_toplu_cure_configuration(self) -> None:
        """Synchronize UI state with Toplu Cure controller."""
        if self._toplu_cure_controller is None:
            return
        config = self._build_toplu_cure_config()
        self._toplu_cure_controller.update_config(config)

    def _make_coordinate_saver(self, row: int, label: QLabel) -> Callable[[], None]:
        """Create a closure that reads and stores the current coordinate."""

        def save_coordinate() -> None:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "common.read_coordinates", "--once"],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                label.setText("HATA")
                return

            output = f"{result.stdout}\n{result.stderr}"
            coord = self._parse_coordinate_output(output)
            if coord is None:
                label.setText("HATA")
                return

            label.setText(f"{coord[0]}, {coord[1]}")
            self._party_coordinates[row] = coord

        return save_coordinate

    def _show_status(self, message: str, timeout: int = 5000) -> None:
        bar = self.statusBar()
        if bar is not None:
            bar.showMessage(message, timeout)

    def _click_monster_target(self, row: int) -> bool:
        if pyautogui is None:
            self._show_status('pyautogui kurulu değil; tıklama yapılamadı.')
            return False
        if self._anti_afk_template is None:
            self._show_status('Şablon yüklenemedi; tıklama yapılamadı.')
            return False

        combo = self._party_offset_combos.get(row)
        offset_cm = 3
        if combo is not None:
            try:
                offset_cm = int(combo.currentText())
            except ValueError:
                offset_cm = 3

        try:
            frame = np.array(pyautogui.screenshot())
        except Exception as exc:
            self._show_status(f'Ekran görüntüsü alınamadı: {exc}')
            return False

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bbox = monster_detector.detect_in_frame(frame_bgr, self._anti_afk_template)
        if not bbox:
            self._show_status('[ANTI AFK] tespit edilemedi.', 3000)
            return False

        x, y, w, h = bbox
        click_x = x + w // 2
        click_y = y + h // 2 + int(offset_cm * CM_TO_PX)

        try:
            pyautogui.click(click_x, click_y)
        except Exception as exc:
            self._show_status(f'Tıklama başarısız: {exc}')
            return False

        self._show_status(f'Tıklama: ({click_x}, {click_y}) [+{offset_cm} cm]', 3000)
        if self._parazit_controller is not None:
            self._parazit_controller.handle_partyms_trigger()
        return True

    def _start_monster_monitor(self, row: int) -> None:
        label = self.findChild(QLabel, f"PartyMsRow{row}StatusLabel")
        if label is None:
            return
        if self._anti_afk_template is None or pyautogui is None:
            label.setText("Şablon/tıklama modülü eksik")
            return
        self._stop_monitor_cycle()
        label.setText("HP: 10 sn sonra kontrol başlayacak")
        self._monitor_row = row
        self._monitor_label = label
        self._monitor_failures = 0
        self._monitor_delay_timer.start(10000)

    def _begin_monitor_checks(self) -> None:
        if self._monitor_row is None:
            return
        label = self._monitor_label
        if label is not None:
            label.setText("HP: kontrol başladı")
        self._monitor_failures = 0
        self._monitor_tick()
        self._monitor_timer.start()

    def _monitor_tick(self) -> None:
        if self._monitor_row is None:
            self._monitor_timer.stop()
            return
        row = self._monitor_row
        label = self._monitor_label or self.findChild(QLabel, f"PartyMsRow{row}StatusLabel")
        detected = self._detect_antiafk()

        if detected:
            self._monitor_failures = 0
            if label is not None:
                label.setText("HP: Yaşıyor")
            return

        self._monitor_failures += 1
        if label is not None:
            label.setText(f"HP: Arama {self._monitor_failures}/10 - bulunamadı")
        if self._monitor_failures >= 10:
            if label is not None:
                label.setText("HP: Ölü")
            should_advance = self._monitor_should_advance
            self._stop_monitor_cycle()
            self._current_navigation_row = None
            if should_advance:
                self._advance_navigation()
                
    def _detect_antiafk(self) -> bool:
        if pyautogui is None or self._anti_afk_template is None:
            return False
        try:
            frame = np.array(pyautogui.screenshot())
        except Exception:
            return False
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return monster_detector.detect_in_frame(frame_bgr, self._anti_afk_template) is not None

    def _stop_monitor_cycle(self) -> None:
        self._monitor_delay_timer.stop()
        self._monitor_timer.stop()
        self._monitor_row = None
        self._monitor_label = None
        self._monitor_failures = 0
        self._monitor_should_advance = False
    def _parse_coordinate_output(self, output: str) -> Optional[Tuple[int, int]]:
        """Extract coordinate pair from command output."""
        numbers = re.findall(r"-?\d+", output)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
        return None

    def _launch_direct_navigation(self, row: int) -> None:
        """Trigger navigation for a single row without sequencing."""
        if row not in self._party_coordinates:
            self._show_status("Önce bu satırın koordinatını kaydedin.", 4000)
            return
        self._stop_navigation()
        self._navigation_queue.clear()
        self._start_navigation_process(row, monitor_after=False, auto_click=True)
        self._show_status("Koordinata gidiliyor...", 2000)

    def _start_navigation_sequence(self) -> None:
        """Begin automated navigation across saved coordinates."""
        if self._paused:
            self._paused = False
            if self._navigation_queue:
                self._advance_navigation()
            return
        if not self._party_coordinates:
            return
        if self._navigation_process and self._navigation_process.state() != QProcess.NotRunning:
            return
        sorted_rows = sorted(self._party_coordinates.keys())
        if not sorted_rows:
            return
        self._paused = False
        self._navigation_queue = sorted_rows
        self._advance_navigation()

    def _advance_navigation(self) -> None:
        """Start navigation for the next row in the queue."""
        if self._navigation_process and self._navigation_process.state() != QProcess.NotRunning:
            return
        self._status_timer.stop()
        self._status_checks = 0
        self._stop_monitor_cycle()
        if not self._navigation_queue:
            self._current_navigation_row = None
            return
        next_row = self._navigation_queue.pop(0)
        self._start_navigation_process(next_row, monitor_after=True, auto_click=True)

    def _start_navigation_process(self, row: int, monitor_after: bool, auto_click: bool = False) -> None:
        """Spawn a navigation process for the given row."""
        target = self._party_coordinates.get(row)
        if not target:
            self._show_status("Bu satır için koordinat kaydı bulunamadı.", 4000)
            return

        if self._navigation_process and self._navigation_process.state() != QProcess.NotRunning:
            self._navigation_process.kill()

        process = QProcess(self)
        process.setWorkingDirectory(str(PROJECT_ROOT))
        process.finished.connect(self._on_navigation_finished)
        process.start(
            sys.executable,
            ["-m", "common.read_coordinates", "--navigate", str(target[0]), str(target[1])],
        )

        self._navigation_process = process
        self._current_navigation_row = row
        self._monitor_after_completion = monitor_after
        self._monitor_should_advance = monitor_after
        self._pending_auto_click_row = row if auto_click else None
        self._status_timer.stop()
        self._status_checks = 0
        self._stop_monitor_cycle()

    def _on_navigation_finished(self) -> None:
        """Handle completion of a navigation subprocess."""
        finished_row = self._current_navigation_row
        self._navigation_process = None
        auto_row = self._pending_auto_click_row
        self._pending_auto_click_row = None
        click_success = False
        if auto_row is not None and auto_row == finished_row:
            click_success = self._click_monster_target(auto_row)
            if click_success:
                self._start_monster_monitor(auto_row)
        if not click_success:
            self._stop_monitor_cycle()
            if self._monitor_after_completion and self._navigation_queue:
                self._current_navigation_row = None
                self._advance_navigation()
            else:
                self._current_navigation_row = None
        else:
            self._current_navigation_row = finished_row

    def _check_current_target_status(self) -> None:
        """Legacy status kontrolü (kullanılmıyor)."""
        self._status_timer.stop()

    def _pause_navigation(self) -> None:
        """Pause the current navigation sequence."""
        if self._navigation_process and self._navigation_process.state() != QProcess.NotRunning:
            self._navigation_process.kill()
            self._navigation_process = None
            if self._current_navigation_row is not None:
                self._navigation_queue.insert(0, self._current_navigation_row)
        self._current_navigation_row = None
        self._status_timer.stop()
        self._pending_auto_click_row = None
        self._stop_monitor_cycle()
        self._paused = True

    def _stop_navigation(self, clear_queue: bool = True) -> None:
        """Stop navigation and optionally clear the pending queue."""
        if self._navigation_process and self._navigation_process.state() != QProcess.NotRunning:
            self._navigation_process.kill()
        self._navigation_process = None
        self._status_timer.stop()
        self._current_navigation_row = None
        self._pending_auto_click_row = None
        self._stop_monitor_cycle()
        if clear_queue:
            self._navigation_queue.clear()
        self._paused = False

    def _reset_navigation(self) -> None:
        """Stop navigation and clear stored coordinates."""
        self._stop_navigation()
        self._stop_monitor_cycle()
        self._party_coordinates.clear()
        for row_index in range(1, 18):
            coord_label = self.findChild(QLabel, f"PartyMsRow{row_index}CoordLabel")
            if coord_label:
                coord_label.setText("--")

    def _on_job_selected(self, button_id: int) -> None:
        """Switch stacked widget to selected job and update header."""
        if button_id < 0 or button_id >= len(self._jobs):
            return
        if self.job_stack:
            self.job_stack.setCurrentIndex(button_id)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        """Ensure background controllers are stopped before closing."""
        if self._cure_controller is not None:
            self._cure_controller.shutdown()
        if self._undy_controller is not None:
            self._undy_controller.shutdown()
        if self._ac_controller is not None:
            self._ac_controller.shutdown()
        if self._mana_controller is not None:
            self._mana_controller.shutdown()
        if self._heal_controller is not None:
            self._heal_controller.shutdown()
        if self._str30_controller is not None:
            self._str30_controller.shutdown()
        if self._zehir_controller is not None:
            self._zehir_controller.shutdown()
        if self._rsiz_controller is not None:
            self._rsiz_controller.shutdown()
        if self._malice_controller is not None:
            self._malice_controller.shutdown()
        self._unregister_malice_from_parazit()
        if self._tekli_controller is not None:
            self._tekli_controller.shutdown()
        self._unregister_tekli_from_malice()
        if self._subside_controller is not None:
            self._subside_controller.shutdown()
        if self._torment_controller is not None:
            self._torment_controller.shutdown()
        if self._parazit_controller is not None:
            self._parazit_controller.shutdown()
        if self._rr_controller is not None:
            self._rr_controller.shutdown()
        if self._toplu_ac_controller is not None:
            self._toplu_ac_controller.shutdown()
        if self._toplu_cure_controller is not None:
            self._toplu_cure_controller.shutdown()
        if self._toplu_buff_controller is not None:
            self._toplu_buff_controller.shutdown()
        if self._restore_controller is not None:
            self._restore_controller.shutdown()
        if self._toplu_controller is not None:
            self._toplu_controller.shutdown()
        super().closeEvent(event)


__all__ = ["MainWindow"]
