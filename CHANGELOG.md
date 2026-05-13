# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- Added adjacent-run intra-PSS estimation alongside the original cumulative intra-PSS route.
- Added `score_intra_annotations()` for post hoc rescoring from saved annotation tables.
- Added `summarize_intra_scores()`, `extract_intra_score_map()`, `extract_inter_score_map()`, and `summarize_inter_scores()` for downstream diagnostics and reporting.

### Changed
- Updated `intra_pss()` to support multiple intra analysis modes while preserving backward-compatible cumulative behavior by default.
- Moved paraphrase-model loading to lazy initialization so post hoc scoring helpers do not require generation models at import time.

### Fixed
- Made API-key-dependent tests defer key lookup until runtime.
- Made the lightweight inter-PSS unit test run without downloading paraphrase models.

## [0.1.4] - 2025-02-26
### Added
- Changed get_openai_api_key() to the more flexible get_api_key() which works with
openai, mistral, anthropic (claude), cohere and huggingface
- Added docstring documentation in core.py
- Added API Documentation link
- Updated README.md to include interoperable examples (OpenAI and Ollama)
- Added CHANGELOG.md file to document changes in each version of the package


### Fixed
- Fixed `parse_function`, so that it can be called within both `intra_pss`
and `inter_pss`
