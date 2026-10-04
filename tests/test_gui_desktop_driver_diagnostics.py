"""A clipped real Tk control must never receive a click aimed at another panel."""

from pathlib import Path
import importlib.util
import os
import sys
import tkinter as tk
from tkinter import ttk
from unittest.mock import Mock

import pytest


@pytest.fixture
def driver():
    if sys.platform != "win32" and not os.environ.get("DISPLAY"):
        pytest.skip("Tk geometry diagnostics require a desktop display.")
    path = Path(__file__).with_name("gui_desktop_driver.py")
    spec = importlib.util.spec_from_file_location("desktop_diagnostics_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coordinate_click_rejects_horizontally_clipped_control(driver, monkeypatch):
    monkeypatch.setattr(driver.sys, "platform", "win32")
    monkeypatch.setenv("GITHUB_ACTIONS", "false")
    root = tk.Tk()
    try:
        root.geometry("600x300+0+0")
        root.lift()
        root.attributes("-topmost", True)
        root.focus_force()
        canvas = tk.Canvas(root, width=180, height=220)
        canvas.pack(side="left")
        interior = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=interior, anchor="nw", width=180)
        ttk.Button(interior, text="First", width=20).grid(row=0, column=0)
        clipped = ttk.Button(interior, text="Fit calibration", width=20)
        clipped.grid(row=0, column=1)
        ttk.Frame(root, width=400, height=220).pack(side="left")
        root.update()
        backend = Mock()
        with pytest.raises(LookupError, match="clipped or occluded"):
            driver._click_widget(root, backend, clipped)
        backend.click.assert_not_called()
    finally:
        root.destroy()


def test_coordinate_click_dispatches_to_visible_control(driver, monkeypatch):
    monkeypatch.setattr(driver.sys, "platform", "win32")
    monkeypatch.setenv("GITHUB_ACTIONS", "false")
    root = tk.Tk()
    try:
        root.geometry("600x300+0+0")
        root.lift()
        root.attributes("-topmost", True)
        root.focus_force()
        button = ttk.Button(root, text="Visible")
        button.pack()
        root.update()
        backend = Mock()
        driver._click_widget(root, backend, button)
        backend.click.assert_called_once_with(*driver._center_of(button))
    finally:
        root.destroy()
