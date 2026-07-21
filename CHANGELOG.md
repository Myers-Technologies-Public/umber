# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it
reaches a first tagged release.

## [Unreleased]

### Added
- Windows support: platform-split clipboard backend (arboard native backends on
  Windows/macOS, `wayland-data-control` on Linux), ConPTY terminals via
  `%ComSpec%`, and `%APPDATA%`-based config resolution.

## [0.1.0-alpha.1] - 2026-07-21

First tagged pre-release. Research-phase build; expect rough edges.

### Added
- Windows support: platform-split clipboard backend (arboard native backends on
  Windows/macOS, `wayland-data-control` on Linux), ConPTY terminals via
  `%ComSpec%`, and `%APPDATA%`-based config resolution.
- Open-source release scaffolding: dual MIT/Apache-2.0 licensing, CI workflow
  (Linux + Windows), and a tag-driven release workflow.
- Installers: Windows MSI (WiX Toolset via cargo-wix) with a Start Menu shortcut
  + PATH entry, and a self-contained Linux AppImage (linuxdeploy/appimagetool);
  both are published to the Releases page alongside portable archives. A
  placeholder brand icon ships under `assets/` (swap `assets/icon.svg` and
  regenerate per `assets/README.md`).
