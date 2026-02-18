from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt

VERSION = "1.0.0"


class AboutDialog(QDialog):
    """Simple About dialog showing app name, version, and license."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Talon")
        self.setFixedSize(420, 300)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(6)

        layout.addSpacing(16)

        # App name
        name_label = QLabel("Talon Assistant")
        name_label.setObjectName("about_app_name")
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(name_label)

        # Version
        version_label = QLabel(f"Version {VERSION}")
        version_label.setObjectName("about_version")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label)

        layout.addSpacing(12)

        # Description
        desc_label = QLabel(
            "A local-first desktop AI assistant with voice control,\n"
            "smart home integration, and a talent plugin system."
        )
        desc_label.setObjectName("about_description")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        layout.addSpacing(16)

        # License
        license_label = QLabel("Licensed under GNU General Public License v3.0")
        license_label.setObjectName("about_license")
        license_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(license_label)

        layout.addSpacing(4)

        # GitHub link
        link_label = QLabel(
            '<a href="https://github.com/talon-assistant/talon-assistant">'
            'github.com/talon-assistant/talon-assistant</a>'
        )
        link_label.setObjectName("about_link")
        link_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        link_label.setOpenExternalLinks(True)
        layout.addWidget(link_label)

        layout.addStretch()

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
