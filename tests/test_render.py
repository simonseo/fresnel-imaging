from __future__ import annotations

from fresnel_imaging.simulation import render


def test_find_blender_prefers_mac_app_bundle(monkeypatch):
    app_blender = "/Applications/Blender.app/Contents/MacOS/Blender"

    monkeypatch.setattr(render.sys, "platform", "darwin")
    monkeypatch.setattr(render.shutil, "which", lambda _: "/Users/sseo/bin/blender")
    monkeypatch.setattr(render.os.path, "isfile", lambda p: p == app_blender)

    assert render._find_blender() == app_blender


def test_find_blender_uses_which_when_no_mac_bundle(monkeypatch):
    monkeypatch.setattr(render.sys, "platform", "darwin")
    monkeypatch.setattr(render.os.path, "isfile", lambda _: False)
    monkeypatch.setattr(render.shutil, "which", lambda _: "/Users/sseo/bin/blender")

    assert render._find_blender() == "/Users/sseo/bin/blender"


def test_find_blender_non_darwin_prefers_which(monkeypatch):
    monkeypatch.setattr(render.sys, "platform", "linux")
    monkeypatch.setattr(render.shutil, "which", lambda _: "/usr/local/bin/blender")

    assert render._find_blender() == "/usr/local/bin/blender"
