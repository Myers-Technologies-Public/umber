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
- Open-source release scaffolding: dual MIT/Apache-2.0 licensing, CI workflow
  (Linux + Windows), and a tag-driven release workflow.
