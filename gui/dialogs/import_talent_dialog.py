import ast
import os
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QFrame)
from PyQt6.QtCore import Qt


class ImportTalentDialog(QDialog):
    """Dialog for importing a custom talent .py file.

    Uses AST-based validation (no execution of untrusted code) to verify
    the file contains a BaseTalent subclass before importing.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_filepath = None
        self.setWindowTitle("Import Talent")
        self.setMinimumSize(500, 350)

        layout = QVBoxLayout(self)

        # Instructions
        info = QLabel(
            "Select a Python file (.py) containing a BaseTalent subclass.\n"
            "The file will be copied to talents/user/ and loaded immediately.")
        info.setWordWrap(True)
        layout.addWidget(info)

        # File picker
        pick_btn = QPushButton("Choose File...")
        pick_btn.clicked.connect(self._pick_file)
        layout.addWidget(pick_btn)

        # File path label
        self.path_label = QLabel("No file selected")
        self.path_label.setObjectName("talent_description")
        layout.addWidget(self.path_label)

        # Validation result panel
        self.info_panel = QFrame()
        self.info_panel.setObjectName("talent_info_panel")
        self.info_panel.setVisible(False)
        info_layout = QVBoxLayout(self.info_panel)

        self.name_label = QLabel()
        self.name_label.setObjectName("talent_validation_success")
        info_layout.addWidget(self.name_label)

        self.desc_label = QLabel()
        self.desc_label.setWordWrap(True)
        info_layout.addWidget(self.desc_label)

        self.keywords_label = QLabel()
        self.keywords_label.setObjectName("talent_keywords")
        self.keywords_label.setWordWrap(True)
        info_layout.addWidget(self.keywords_label)

        layout.addWidget(self.info_panel)

        # Error label
        self.error_label = QLabel()
        self.error_label.setObjectName("talent_validation_error")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.import_btn = QPushButton("Import")
        self.import_btn.setEnabled(False)
        self.import_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.import_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _pick_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Talent File", "", "Python Files (*.py)")
        if filepath:
            self.path_label.setText(filepath)
            self._validate_file(filepath)

    def _validate_file(self, filepath):
        """Parse the .py file with AST to find BaseTalent subclasses."""
        self.info_panel.setVisible(False)
        self.error_label.setVisible(False)
        self.import_btn.setEnabled(False)
        self.selected_filepath = None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            tree = ast.parse(source)
        except SyntaxError as e:
            self._show_error(f"Syntax error in file: {e}")
            return
        except Exception as e:
            self._show_error(f"Cannot read file: {e}")
            return

        # Check for existing file in talents/user/
        filename = os.path.basename(filepath)
        dest = os.path.join("talents", "user", filename)
        if os.path.exists(dest):
            self._show_error(
                f"A talent file named '{filename}' already exists in "
                f"talents/user/. Remove it first or rename your file.")
            return

        # Find BaseTalent subclasses
        talent_classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = ""
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = base.attr
                    if base_name == "BaseTalent":
                        info = self._extract_talent_info(node)
                        talent_classes.append(info)

        if not talent_classes:
            self._show_error(
                "No BaseTalent subclass found in this file.\n"
                "The talent class must inherit from BaseTalent.")
            return

        # Show the first talent class found
        talent = talent_classes[0]
        self.name_label.setText(f"Name: {talent['name'] or '(not set)'}")
        self.desc_label.setText(
            f"Description: {talent['description'] or '(not set)'}")
        kw = ", ".join(talent["keywords"]) if talent["keywords"] else "(none)"
        self.keywords_label.setText(f"Keywords: {kw}")
        self.info_panel.setVisible(True)

        if len(talent_classes) > 1:
            self.desc_label.setText(
                self.desc_label.text() +
                f"\n\n({len(talent_classes)} talent classes found; "
                f"all will be imported)")

        self.selected_filepath = filepath
        self.import_btn.setEnabled(True)

    def _extract_talent_info(self, class_node):
        """Extract name, description, keywords from AST class body."""
        info = {"name": "", "description": "", "keywords": []}
        for stmt in class_node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        if (target.id == "name"
                                and isinstance(stmt.value, ast.Constant)):
                            info["name"] = stmt.value.value
                        elif (target.id == "description"
                              and isinstance(stmt.value, ast.Constant)):
                            info["description"] = stmt.value.value
                        elif (target.id == "keywords"
                              and isinstance(stmt.value, ast.List)):
                            info["keywords"] = [
                                elt.value for elt in stmt.value.elts
                                if isinstance(elt, ast.Constant)
                            ]
        return info

    def _show_error(self, msg):
        self.error_label.setText(msg)
        self.error_label.setVisible(True)
        self.info_panel.setVisible(False)
        self.import_btn.setEnabled(False)
        self.selected_filepath = None
