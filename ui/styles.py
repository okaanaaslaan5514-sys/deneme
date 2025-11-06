"""Temel QSS stilleri LUX kontrol arayüzü için."""

BASE_QSS = """
QWidget {
    background-color: #050608;
    color: #f5d742;
    font-family: 'Segoe UI', 'Roboto', sans-serif;
    font-size: 14px;
}

QFrame#NavigationFrame {
    background: transparent;
}

QLabel#BrandTitleLabel {
    font-size: 38px;
    font-weight: 700;
    letter-spacing: 6px;
    color: #f5d742;
}

QLabel#BrandSubtitleLabel {
    font-size: 16px;
    letter-spacing: 3px;
    color: #f1c40f;
}

QFrame#NavigationFrame QToolButton {
    border: none;
    border-radius: 12px;
    padding: 12px 16px;
    text-align: left;
    color: #f5d742;
    background: transparent;
    font-size: 16px;
    font-weight: 600;
}

QFrame#NavigationFrame QToolButton:hover {
    background-color: rgba(245, 215, 66, 0.18);
}

QFrame#NavigationFrame QToolButton:checked {
    background-color: rgba(245, 215, 66, 0.32);
}

QFrame#SupportAttackFrame,
QFrame#PartyFrame {
    background-color: #0f1115;
    border: 1px solid #1f232b;
    border-radius: 16px;
    padding: 24px;
}

QLabel#SupportHeaderLabel,
QLabel#AttackHeaderLabel,
QLabel#PartySectionHeader {
    font-weight: 700;
    letter-spacing: 1.6px;
    font-size: 16px;
    color: #f8e27c;
    margin-bottom: 12px;
}

QLabel#PartyRestoreHeader,
QLabel#PartyTopluHeader {
    font-size: 12px;
    font-weight: 600;
    color: #f5d742;
    padding: 6px 0;
}

QCheckBox {
    spacing: 10px;
    font-weight: 500;
}

QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border-radius: 12px;
    border: 2px solid #f5d742;
    background-color: #050608;
}

QCheckBox::indicator:checked {
    background-color: #f5d742;
}

QLabel {
    color: #f5d742;
}

QComboBox {
    background-color: #161922;
    border: 1px solid #2c3140;
    border-radius: 10px;
    padding: 6px 12px;
    color: #f5d742;
    min-width: 110px;
}

QComboBox QAbstractItemView {
    background-color: #161922;
    color: #f5d742;
    border: 1px solid #2c3140;
    selection-background-color: #2c3140;
    selection-color: #f5d742;
}

QComboBox QAbstractItemView::item {
    padding: 6px 12px;
    color: #f5d742;
}

QComboBox::drop-down {
    border: none;
}

QComboBox:on {
    border-color: #f5d742;
}

QPushButton {
    background-color: #161922;
    border: 1px solid #2c3140;
    border-radius: 10px;
    padding: 8px 14px;
    color: #f5d742;
    font-weight: 600;
}

QPushButton:hover {
    border-color: #f5d742;
}
"""
