"""
Unit tests for STAF Preprocessing Stage 4: Audio Extraction.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from staf.preprocessing.extract_audio import (
    extract_audio_from_video,
    run_audio_extraction_pipeline,
)


@pytest.fixture
def temp_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@patch("subprocess.run")
def test_extract_audio_from_video_success(mock_run: MagicMock, temp_dir: Path) -> None:
    """Tests a successful audio extraction with mocked subprocess."""
    video_path = temp_dir / "test_video.mp4"
    # Touch the video path to simulate its existence
    video_path.touch()
    
    output_dir = temp_dir / "audio"
    target_audio = output_dir / "test_video.wav"

    # Configure mock to simulate successful ffmpeg run
    mock_response = MagicMock()
    mock_response.returncode = 0
    mock_run.returndict = {"returncode": 0}
    mock_run.return_value = mock_response

    # We must patch Path.exists/Path.stat to simulate ffmpeg creating a non-empty output file
    # during the verification step.
    original_exists = Path.exists
    original_stat = Path.stat
    
    def mock_exists(self: Path) -> bool:
        if self == target_audio:
            return True
        return original_exists(self)

    class MockStat:
        st_size = 1024

    def mock_stat(self: Path) -> MockStat:
        if self == target_audio:
            return MockStat()
        return original_stat(self)

    with patch.object(Path, "exists", autospec=True, side_effect=mock_exists), \
         patch.object(Path, "stat", autospec=True, side_effect=mock_stat):
        
        success, msg = extract_audio_from_video(
            video_path=video_path,
            output_dir=output_dir,
            sample_rate=16000,
            overwrite=True,
        )

    assert success is True
    assert "SUCCESS" in msg
    
    # Assert ffmpeg was called with expected arguments
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == "ffmpeg"
    assert "-ar" in args
    assert "16000" in args
    assert "-ac" in args
    assert "1" in args
    assert str(target_audio) in args


@patch("subprocess.run")
def test_extract_audio_skip_existing(mock_run: MagicMock, temp_dir: Path) -> None:
    """Verifies that extraction is skipped if the target audio already exists and overwrite is False."""
    video_path = temp_dir / "test_video.mp4"
    video_path.touch()
    
    output_dir = temp_dir / "audio"
    output_dir.mkdir()
    
    target_audio = output_dir / "test_video.wav"
    target_audio.write_text("dummy audio content")

    success, msg = extract_audio_from_video(
        video_path=video_path,
        output_dir=output_dir,
        overwrite=False,
    )

    assert success is True
    assert "SKIPPED" in msg
    mock_run.assert_not_called()


@patch("subprocess.run")
def test_extract_audio_ffmpeg_failure(mock_run: MagicMock, temp_dir: Path) -> None:
    """Verifies that ffmpeg errors are handled correctly."""
    video_path = temp_dir / "test_video.mp4"
    video_path.touch()
    
    output_dir = temp_dir / "audio"

    # Configure mock to simulate failed ffmpeg run
    mock_response = MagicMock()
    mock_response.returncode = 1
    mock_response.stderr = "ffmpeg mock error description"
    mock_run.return_value = mock_response

    success, msg = extract_audio_from_video(
        video_path=video_path,
        output_dir=output_dir,
        overwrite=True,
    )

    assert success is False
    assert "FAILED: ffmpeg error" in msg
