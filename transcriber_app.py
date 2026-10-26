"""Transcribe a recording that stays on this PC.

The recording is never uploaded. Hugging Face is contacted only when the
chosen Whisper model is not already in the local cache.
"""

from __future__ import annotations

# Windows must be told this before Tk loads, or the window is drawn smaller
# than the layout and the buttons are cut off.
def _enable_windows_dpi() -> None:
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


_enable_windows_dpi()

import os
import re
import subprocess
import tempfile
import threading
import tkinter as tk
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable

# The model library can phone home with usage statistics. This app does not.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

SAMPLE_RATE = 16000
PCM_SCALE = 1.0 / 32768.0
LOW_CONFIDENCE = 0.6
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v3")

# Display name, Whisper language code. None means detect the language.
LANGUAGES: tuple[tuple[str, str | None], ...] = (
    ("Auto", None),
    ("English", "en"),
    ("Sinhala", "si"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Hindi", "hi"),
    ("Tamil", "ta"),
    ("Chinese", "zh"),
    ("Arabic", "ar"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Italian", "it"),
)

# Names for languages Auto may detect that are not in the short menu.
LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Portuguese",
    "tr": "Turkish",
    "pl": "Polish",
    "nl": "Dutch",
    "ar": "Arabic",
    "sv": "Swedish",
    "it": "Italian",
    "id": "Indonesian",
    "hi": "Hindi",
    "fi": "Finnish",
    "vi": "Vietnamese",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "el": "Greek",
    "ms": "Malay",
    "cs": "Czech",
    "ro": "Romanian",
    "da": "Danish",
    "hu": "Hungarian",
    "ta": "Tamil",
    "no": "Norwegian",
    "th": "Thai",
    "ur": "Urdu",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "lt": "Lithuanian",
    "si": "Sinhala",
    "bn": "Bengali",
    "fa": "Persian",
    "te": "Telugu",
    "mr": "Marathi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "gu": "Gujarati",
    "ne": "Nepali",
    "my": "Myanmar",
    "km": "Khmer",
    "lo": "Lao",
    "ka": "Georgian",
    "am": "Amharic",
    "yue": "Cantonese",
}

FILE_TYPES = (
    ("Audio and video", "*.mp3;*.m4a;*.wav;*.mp4;*.webm"),
    ("MP3 audio", "*.mp3"),
    ("M4A audio", "*.m4a"),
    ("WAV audio", "*.wav"),
    ("MP4 video", "*.mp4"),
    ("WebM video", "*.webm"),
)

DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
OUT_TIME_RE = re.compile(r"out_time=(\d+):(\d+):(\d+(?:\.\d+)?)")
STREAM_RE = re.compile(r"Stream\s+#\d+:\d+")

DOWNLOAD_ERROR_NAMES = {
    "LocalEntryNotFoundError",
    "HfHubHTTPError",
    "GatedRepoError",
    "RepositoryNotFoundError",
    "EntryNotFoundError",
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "ConnectionError",
    "TimeoutError",
    "ProtocolError",
    "SSLError",
    "URLError",
    "HTTPError",
    "ProxyError",
}


class Cancelled(Exception):
    """The user stopped the current transcription."""


class MissingFileError(Exception):
    """The chosen recording is not on disk."""


class NoAudioError(Exception):
    """The file does not contain an audio stream."""


class MediaError(Exception):
    """The recording could not be prepared for transcription."""


class ModelLoadError(Exception):
    """The Whisper model could not be loaded or downloaded."""


@dataclass(frozen=True)
class MediaInfo:
    duration: float | None
    has_audio: bool
    is_video: bool


@dataclass(frozen=True)
class Piece:
    start: float
    end: float
    text: str


class ModelCache:
    """Keep the last CPU model loaded so a second run does not reload it."""

    def __init__(self) -> None:
        self.name: str | None = None
        self.model: object | None = None

    def get(self, name: str) -> object:
        if self.model is not None and self.name == name:
            return self.model
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(name, device="cpu", compute_type="int8")
        except Exception as exc:
            raise ModelLoadError(str(exc)) from exc
        self.model = model
        self.name = name
        return model


class ProcessBox:
    """Lets the window stop the ffmpeg process started by the worker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.process: subprocess.Popen[bytes] | None = None

    def set(self, process: subprocess.Popen[bytes] | None) -> None:
        with self._lock:
            self.process = process

    def stop(self) -> None:
        with self._lock:
            process = self.process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def format_clock(seconds: float) -> str:
    """Hour-minute-second clock, rounded to the nearest second."""
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def language_label(code: str, *, detected: bool) -> str:
    name = LANGUAGE_NAMES.get(code, code)
    label = f"{name} ({code})"
    if detected:
        return f"{label}, detected automatically"
    return label


def check_cancel(cancel_event: threading.Event) -> None:
    if cancel_event.is_set():
        raise Cancelled()


def _run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW,
        check=False,
    )


def probe_media(path: Path) -> MediaInfo:
    """Read duration and whether the file has audio, without decoding it all."""
    if not path.exists():
        raise MissingFileError(str(path))
    if not path.is_file():
        raise MediaError("Choose a recording file, not a folder.")

    result = _run_ffmpeg(
        [ffmpeg_exe(), "-hide_banner", "-nostats", "-i", str(path)]
    )
    text = result.stderr.decode("utf-8", errors="replace")
    lowered = text.lower()
    if not path.is_file() or "no such file or directory" in lowered:
        raise MissingFileError(str(path))

    has_audio = False
    is_video = False
    for line in text.splitlines():
        if not STREAM_RE.search(line):
            continue
        if re.search(r"\bAudio\s*:", line):
            has_audio = True
        if re.search(r"\bVideo\s*:", line) and not re.search(
            r"Video\s*:\s*(mjpeg|png)\b", line, re.IGNORECASE
        ):
            is_video = True

    match = DURATION_RE.search(text)
    duration = None
    if match:
        hours, minutes, seconds = match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    if not has_audio and not is_video and result.returncode != 0:
        raise MediaError(_unreadable_message(text))
    return MediaInfo(duration=duration, has_audio=has_audio, is_video=is_video)


def _unreadable_message(ffmpeg_text: str) -> str:
    reason = ""
    for line in reversed(ffmpeg_text.splitlines()):
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned.lower().startswith("error opening input"):
            continue
        reason = cleaned
        break
    if reason.lower().startswith("error opening input files:"):
        reason = reason.split(":", 1)[-1].strip()
    if len(reason) > 240:
        reason = reason[:240].rstrip() + "…"
    if reason:
        return (
            "The recording could not be read. "
            "Choose an mp3, m4a, wav, mp4, or webm file. "
            f"{reason}"
        )
    return (
        "The recording could not be read. "
        "Choose an mp3, m4a, wav, mp4, or webm file."
    )


def extract_wav(
    source: Path,
    wav_path: Path,
    duration: float | None,
    *,
    cancel_event: threading.Event,
    on_progress: Callable[[float | None], None],
    processes: ProcessBox,
) -> None:
    """Write 16 kHz mono PCM so the model never has to open the recording."""
    command = [
        ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-nostats",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        "-progress",
        "pipe:1",
        str(wav_path),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW,
    )
    processes.set(process)
    stderr_parts: list[bytes] = []

    def drain_stderr() -> None:
        if process.stderr is None:
            return
        try:
            stderr_parts.append(process.stderr.read())
        except OSError:
            pass

    drainer = threading.Thread(target=drain_stderr, daemon=True)
    drainer.start()
    code = 1
    try:
        assert process.stdout is not None
        while True:
            check_cancel(cancel_event)
            line = process.stdout.readline()
            if not line:
                break
            match = OUT_TIME_RE.search(line.decode("utf-8", errors="replace"))
            if match and duration and duration > 0:
                hours, minutes, seconds = match.groups()
                current = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                fraction = min(1.0, current / duration)
                on_progress(5 + 30 * fraction)
        code = process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        processes.set(None)
        drainer.join(timeout=2)

    check_cancel(cancel_event)
    if code != 0 or not wav_path.is_file():
        detail = b"".join(stderr_parts).decode("utf-8", errors="replace")
        lowered = detail.lower()
        if "no such file or directory" in lowered or not source.is_file():
            raise MissingFileError(str(source))
        if (
            "does not contain any stream" in lowered
            or "output file #0 does not contain any stream" in lowered
            or "no audio" in lowered
        ):
            raise NoAudioError(str(source))
        raise MediaError(_unreadable_message(detail))


def read_wav_float32(wav_path: Path):
    """Load the converted WAV as float32 samples scaled by 1/32768."""
    import numpy as np

    with wave.open(str(wav_path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if channels != 1 or width != 2 or rate != SAMPLE_RATE:
        raise MediaError("The converted audio was not 16 kHz mono.")
    if not frames:
        raise NoAudioError(str(wav_path))
    pcm = np.frombuffer(frames, dtype="<i2")
    audio = pcm.astype(np.float32) * np.float32(PCM_SCALE)
    return np.ascontiguousarray(audio, dtype=np.float32)


def group_pieces(pieces: list[Piece]) -> list[list[Piece]]:
    """Break the transcript into paragraphs at pauses and at long sentences."""
    groups: list[list[Piece]] = []
    current: list[Piece] = []
    current_chars = 0
    for piece in pieces:
        if current:
            gap = piece.start - current[-1].end
            previous = current[-1].text
            pause = gap >= 1.2
            long_sentence = current_chars >= 480 and previous.endswith(
                (".", "!", "?", "…", "।")
            )
            if pause or long_sentence:
                groups.append(current)
                current = []
                current_chars = 0
        current.append(piece)
        current_chars += len(piece.text) + 1
    if current:
        groups.append(current)
    return groups


def build_transcript(
    *,
    source_name: str,
    duration_text: str,
    language_text: str,
    model_name: str,
    local_date: str,
    pieces: list[Piece],
    timestamps: bool,
    low_confidence_note: str | None,
) -> tuple[str, str]:
    """Return the plain-text transcript and the Markdown transcript."""
    fields = (
        ("Source", source_name),
        ("Duration", duration_text),
        ("Language", language_text),
        ("Model", model_name),
        ("Date", local_date),
    )
    if pieces:
        body_groups = group_pieces(pieces)
    else:
        body_groups = []

    text_header = [f"{label}: {value}" for label, value in fields]
    text_parts = ["\n".join(text_header), ""]
    if low_confidence_note:
        text_parts.extend([low_confidence_note, ""])
    if not body_groups:
        text_parts.append("No speech was detected.")
    else:
        blocks = []
        for group in body_groups:
            if timestamps:
                blocks.append(
                    "\n".join(
                        f"[{format_clock(piece.start)}] {piece.text}" for piece in group
                    )
                )
            else:
                blocks.append(" ".join(piece.text for piece in group))
        text_parts.append("\n\n".join(blocks))
    plain = "\n".join(text_parts).rstrip() + "\n"

    markdown_lines = ["# Transcript", ""]
    for label, value in fields:
        if label == "Source":
            safe_name = source_name.replace("`", "'")
            markdown_lines.append(f"- {label}: `{safe_name}`")
        else:
            markdown_lines.append(f"- {label}: {value}")
    markdown_lines.append("")
    if low_confidence_note:
        markdown_lines.extend([low_confidence_note, ""])
    if not body_groups:
        markdown_lines.append("No speech was detected.")
    else:
        blocks = []
        for group in body_groups:
            if timestamps:
                blocks.append(
                    "\n".join(
                        f"**{format_clock(piece.start)}** {piece.text}  "
                        for piece in group
                    )
                )
            else:
                blocks.append(" ".join(piece.text for piece in group))
        markdown_lines.append("\n\n".join(blocks))
    markdown = "\n".join(markdown_lines).rstrip() + "\n"
    return plain, markdown


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def transcribe_recording(
    source: Path,
    output_dir: Path | None,
    timestamps: bool,
    model_name: str,
    language_code: str | None,
    *,
    cancel_event: threading.Event,
    on_status: Callable[[str], None],
    on_progress: Callable[[float | None], None],
    models: ModelCache,
    processes: ProcessBox,
) -> tuple[Path, Path, bool]:
    """Turn one recording into a .txt file and a .md file.

    Returns the two paths and whether automatic language detection was uncertain.
    """
    check_cancel(cancel_event)
    source = source.expanduser()
    if not source.is_file():
        raise MissingFileError(str(source))

    on_status("Checking the recording…")
    on_progress(2)
    info = probe_media(source)
    if not info.has_audio:
        raise NoAudioError(str(source))
    check_cancel(cancel_event)

    folder = output_dir if output_dir is not None else source.parent
    if not folder.is_dir():
        raise MediaError("The output folder could not be found.")
    stem = source.stem or "transcript"
    txt_path = folder / f"{stem}.txt"
    md_path = folder / f"{stem}.md"
    if txt_path.resolve() == source.resolve() or md_path.resolve() == source.resolve():
        raise MediaError(
            "The transcript would overwrite the recording. Choose another output folder."
        )

    if info.is_video:
        on_status("Extracting audio from the video…")
    else:
        on_status("Converting the audio…")
    on_progress(5)

    fd, raw_temp = tempfile.mkstemp(prefix="local-transcriber-", suffix=".wav")
    os.close(fd)
    wav_path = Path(raw_temp)
    try:
        extract_wav(
            source,
            wav_path,
            info.duration,
            cancel_event=cancel_event,
            on_progress=on_progress,
            processes=processes,
        )
        check_cancel(cancel_event)
        on_status("Reading the audio…")
        on_progress(38)
        audio = read_wav_float32(wav_path)
        duration = float(audio.shape[0]) / SAMPLE_RATE

        on_status(f"Loading the {model_name} model…")
        on_progress(None)
        check_cancel(cancel_event)
        model = models.get(model_name)
        check_cancel(cancel_event)

        on_status("Transcribing the recording…")
        on_progress(45)
        # Samples are passed in memory. Giving transcribe a file path makes it
        # open the file with the installed av package, which fails on this PC.
        segments, transcript_info = model.transcribe(
            audio,
            language=language_code,
            vad_filter=False,
        )

        pieces: list[Piece] = []
        for segment in segments:
            check_cancel(cancel_event)
            # TODO: speaker diarization could be added here later. Whisper does
            # not identify who is speaking, so speaker names must not be invented.
            text = " ".join(segment.text.split())
            if not text:
                continue
            pieces.append(Piece(float(segment.start), float(segment.end), text))
            if duration > 0:
                on_progress(45 + 50 * min(1.0, float(segment.end) / duration))

        detected = language_code is None
        code = transcript_info.language or language_code or "en"
        probability = float(transcript_info.language_probability)
        low_note = None
        uncertain = detected and probability < LOW_CONFIDENCE
        if uncertain:
            low_note = (
                f"Detected language confidence is {probability:.2f}, which is under "
                "0.6. The transcript below is the full text."
            )
        header_duration = format_clock(duration if duration > 0 else (info.duration or 0))
        plain, markdown = build_transcript(
            source_name=source.name,
            duration_text=header_duration,
            language_text=language_label(code, detected=detected),
            model_name=model_name,
            local_date=datetime.now().strftime("%Y-%m-%d"),
            pieces=pieces,
            timestamps=timestamps,
            low_confidence_note=low_note,
        )
        check_cancel(cancel_event)
        on_status("Writing the transcript…")
        on_progress(98)
        try:
            _atomic_write(txt_path, plain)
            _atomic_write(md_path, markdown)
        except OSError as exc:
            raise MediaError(
                "The transcript could not be saved. "
                "Check that the output folder is writable."
            ) from exc
        on_progress(100)
        return txt_path, md_path, uncertain
    finally:
        processes.stop()
        try:
            wav_path.unlink(missing_ok=True)
        except OSError:
            pass


def is_model_download_error(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in DOWNLOAD_ERROR_NAMES:
            return True
        text = str(current).lower()
        if any(
            phrase in text
            for phrase in (
                "huggingface",
                "connection",
                "timed out",
                "timeout",
                "network is unreachable",
                "name resolution",
                "temporary failure",
                "offline",
                "could not download",
                "failed to download",
                "local entry not found",
                "cannot find the requested files",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def friendly_error(exc: BaseException) -> str:
    if isinstance(exc, MissingFileError):
        return "The recording could not be found. Choose the file again."
    if isinstance(exc, NoAudioError):
        return "This file has no audio stream."
    if isinstance(exc, ModelLoadError):
        detail = _short_detail(exc)
        if is_model_download_error(exc):
            message = (
                "The model could not be downloaded. "
                "A model size that is not already on this PC is downloaded once "
                "from Hugging Face. Check the connection and try again."
            )
        else:
            message = "The model could not be loaded."
        if detail:
            return f"{message}\n{detail}"
        return message
    if isinstance(exc, MediaError):
        return str(exc)
    detail = _short_detail(exc)
    if detail:
        return f"Something went wrong.\n{detail}"
    return "Something went wrong."


def _short_detail(exc: BaseException) -> str:
    text = str(exc.__cause__ or exc).strip()
    text = " ".join(text.split())
    if len(text) > 400:
        text = text[:400].rstrip() + "…"
    return text


class TranscriberApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.alive = True
        self.running = False
        self.recording: Path | None = None
        self.output_dir: Path | None = None
        self.cancel_event = threading.Event()
        self.models = ModelCache()
        self.processes = ProcessBox()
        self.language_by_name = {name: code for name, code in LANGUAGES}

        root.title("Local Transcriber")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build()
        self._fit_to_screen()
        root.after(150, self._bring_forward)

    def _build(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 4))
        style.configure("TCheckbutton", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Hint.TLabel", font=("Segoe UI", 9), foreground="#444444")

        outer = ttk.Frame(self.root, padding=(16, 10, 16, 10))
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        ttk.Label(outer, text="Local Transcriber", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            outer,
            text="Turns a recording on this PC into a transcript. The file is not uploaded.",
            style="Hint.TLabel",
            wraplength=640,
        ).grid(row=1, column=0, sticky="w", pady=(0, 8))

        ttk.Label(outer, text="Recording").grid(row=2, column=0, sticky="w")
        recording_row = ttk.Frame(outer)
        recording_row.grid(row=3, column=0, sticky="ew", pady=(2, 4))
        recording_row.columnconfigure(0, weight=1)
        self.recording_entry = ttk.Entry(recording_row)
        self.recording_entry.grid(row=0, column=0, sticky="ew")
        self.recording_entry.configure(state="readonly")
        ttk.Button(
            recording_row, text="Browse", command=self.choose_recording, width=12
        ).grid(row=0, column=1, padx=(8, 0))

        self.file_name_var = tk.StringVar(value="File name: none selected")
        self.duration_var = tk.StringVar(value="Duration: —")
        self.file_name_label = ttk.Label(
            outer, textvariable=self.file_name_var, wraplength=640
        )
        self.file_name_label.grid(row=4, column=0, sticky="w")
        ttk.Label(outer, textvariable=self.duration_var).grid(
            row=5, column=0, sticky="w", pady=(0, 6)
        )

        ttk.Label(outer, text="Output folder").grid(row=6, column=0, sticky="w")
        output_row = ttk.Frame(outer)
        output_row.grid(row=7, column=0, sticky="ew", pady=(2, 2))
        output_row.columnconfigure(0, weight=1)
        self.output_entry = ttk.Entry(output_row)
        self.output_entry.grid(row=0, column=0, sticky="ew")
        self.output_entry.configure(state="readonly")
        ttk.Button(
            output_row, text="Browse", command=self.choose_output_folder, width=12
        ).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(
            output_row,
            text="Save beside the recording",
            command=self.clear_output_folder,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.output_hint = ttk.Label(
            outer,
            text="Leave this empty to save the transcript beside the recording.",
            style="Hint.TLabel",
            wraplength=640,
        )
        self.output_hint.grid(row=8, column=0, sticky="w", pady=(0, 6))

        options = ttk.Frame(outer)
        options.grid(row=9, column=0, sticky="ew", pady=(0, 4))
        self.timestamps_var = tk.BooleanVar(value=True)
        self.timestamps_check = ttk.Checkbutton(
            options,
            text="Include timestamps",
            variable=self.timestamps_var,
        )
        self.timestamps_check.grid(row=0, column=0, sticky="w", padx=(0, 18))
        ttk.Label(options, text="Model size").grid(row=0, column=1, sticky="w")
        self.model_box = ttk.Combobox(
            options,
            values=MODEL_SIZES,
            state="readonly",
            width=12,
        )
        self.model_box.set("small")
        self.model_box.grid(row=0, column=2, sticky="w", padx=(6, 18))
        ttk.Label(options, text="Language").grid(row=0, column=3, sticky="w")
        self.language_box = ttk.Combobox(
            options,
            values=[name for name, _code in LANGUAGES],
            state="readonly",
            width=14,
        )
        self.language_box.set("Auto")
        self.language_box.grid(row=0, column=4, sticky="w", padx=(6, 0))

        ttk.Label(
            outer,
            text="Small is already on this PC. Any other size downloads once.",
            style="Hint.TLabel",
        ).grid(row=10, column=0, sticky="w", pady=(0, 8))

        buttons = ttk.Frame(outer)
        buttons.grid(row=11, column=0, sticky="w", pady=(0, 6))
        self.start_button = ttk.Button(
            buttons, text="Start", command=self.on_start, width=14
        )
        self.start_button.grid(row=0, column=0)
        self.cancel_button = ttk.Button(
            buttons, text="Cancel", command=self.on_cancel, width=14, state="disabled"
        )
        self.cancel_button.grid(row=0, column=1, padx=(8, 0))

        self.progress = ttk.Progressbar(outer, mode="determinate", maximum=100)
        self.progress.grid(row=12, column=0, sticky="ew", pady=(0, 6))

        ttk.Label(outer, text="Status").grid(row=13, column=0, sticky="w")
        self.status = tk.Text(
            outer,
            height=3,
            wrap="word",
            font=("Segoe UI", 10),
            relief="solid",
            borderwidth=1,
            highlightthickness=0,
            padx=6,
            pady=4,
        )
        try:
            self.status.configure(background=self._window_background())
        except tk.TclError:
            pass
        self.status.grid(row=14, column=0, sticky="nsew")
        self.status.tag_configure("info", foreground="#1a1a1a")
        self.status.tag_configure("error", foreground="#8a1c1c")
        self.status.tag_configure("success", foreground="#0b5d0b")
        self.status.bind("<Key>", self._status_key)
        outer.rowconfigure(14, weight=1)
        self._set_status("Ready.", "info")

        self._option_widgets = (
            self.timestamps_check,
            self.model_box,
            self.language_box,
        )
        self.root.bind("<Configure>", self._on_resize)

    def _window_background(self) -> str:
        try:
            return str(self.root.cget("background"))
        except tk.TclError:
            return "SystemButtonFace"

    def _work_area(self) -> tuple[int, int]:
        if os.name == "nt":
            import ctypes

            class RECT(ctypes.Structure):
                _fields_ = (
                    ("left", ctypes.c_long),
                    ("top", ctypes.c_long),
                    ("right", ctypes.c_long),
                    ("bottom", ctypes.c_long),
                )

            rect = RECT()
            if ctypes.windll.user32.SystemParametersInfoW(
                48, 0, ctypes.byref(rect), 0
            ):
                return rect.right - rect.left, rect.bottom - rect.top
        return self.root.winfo_screenwidth(), self.root.winfo_screenheight()

    def _fit_to_screen(self) -> None:
        self.root.update_idletasks()
        req_w = self.root.winfo_reqwidth()
        req_h = self.root.winfo_reqheight()
        work_w, work_h = self._work_area()
        margin = 24
        # Prefer a comfortable size, but stay inside the work area so a
        # 1280x720 display can show every control.
        # Stay inside the work area, so a 1280x720 display can show the window.
        width = min(max(req_w, 780), max(320, work_w - margin))
        height = min(max(req_h, 560), max(240, work_h - margin))
        if req_h > height:
            self.status.configure(height=3)
            self.root.update_idletasks()
            req_h = self.root.winfo_reqheight()
            height = min(max(req_h, 560), max(240, work_h - margin))
        self.root.minsize(min(req_w, width), min(req_h, height))
        self.root.geometry(f"{int(width)}x{int(height)}+24+24")
        self.root.update()
        got_w = self.root.winfo_width()
        got_h = self.root.winfo_height()
        # Some Windows DPI setups shrink the real window. Ask again so the
        # controls are not clipped.
        if got_w + 8 < width or got_h + 8 < height:
            corrected_w = min(int(width * width / max(got_w, 1)), work_w - margin)
            corrected_h = min(int(height * height / max(got_h, 1)), work_h - margin)
            self.root.geometry(f"{corrected_w}x{corrected_h}+24+24")
        self._apply_wrap(int(width))

    def _bring_forward(self) -> None:
        if not self.alive:
            return
        try:
            self.root.lift()
            self.root.focus_force()
        except tk.TclError:
            pass

    def _on_resize(self, event: tk.Event) -> None:
        if event.widget is self.root:
            self._apply_wrap(event.width)

    def _apply_wrap(self, width: int) -> None:
        wrap = max(280, width - 56)
        for child in self.root.winfo_children():
            self._wrap_hints(child, wrap)

    def _wrap_hints(self, widget: tk.Widget, wrap: int) -> None:
        if widget in (getattr(self, "file_name_label", None),):
            try:
                widget.configure(wraplength=wrap)
            except tk.TclError:
                pass
        if isinstance(widget, ttk.Label):
            try:
                if str(widget.cget("style")) == "Hint.TLabel":
                    widget.configure(wraplength=wrap)
            except tk.TclError:
                pass
        for child in widget.winfo_children():
            self._wrap_hints(child, wrap)

    def _ui(self, callback: Callable[[], None]) -> None:
        if not self.alive:
            return
        try:
            self.root.after(0, self._guard, callback)
        except tk.TclError:
            pass

    def _guard(self, callback: Callable[[], None]) -> None:
        if not self.alive:
            return
        try:
            callback()
        except tk.TclError:
            pass

    def _set_entry(self, entry: ttk.Entry, value: str) -> None:
        entry.configure(state="normal")
        entry.delete(0, "end")
        entry.insert(0, value)
        entry.configure(state="readonly")
        entry.xview_moveto(1)

    def _status_key(self, event: tk.Event) -> str | None:
        # The status text stays read-only, and the paths can still be copied.
        if event.state & 0x4 and event.keysym.lower() in {"c", "a"}:
            return None
        return "break"

    def _set_status(self, message: str, kind: str) -> None:
        self.status.configure(state="normal")
        self.status.delete("1.0", "end")
        self.status.insert("1.0", message, kind)
        self.status.configure(state="disabled")

    def _set_progress(self, percent: float | None) -> None:
        if percent is None:
            if str(self.progress.cget("mode")) != "indeterminate":
                self.progress.configure(mode="indeterminate")
            self.progress.start(12)
            return
        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=100)
        self.progress["value"] = max(0.0, min(100.0, percent))

    def _set_running(self, running: bool) -> None:
        self.running = running
        start_state = "disabled" if running else "normal"
        cancel_state = "normal" if running else "disabled"
        browse_state = "disabled" if running else "normal"
        self.start_button.configure(state=start_state)
        self.cancel_button.configure(state=cancel_state)
        for widget in self.root.winfo_children():
            self._set_browse_state(widget, browse_state)
        box_state = "disabled" if running else "readonly"
        self.model_box.configure(state=box_state)
        self.language_box.configure(state=box_state)
        self.timestamps_check.configure(state=start_state)

    def _set_browse_state(self, widget: tk.Widget, state: str) -> None:
        if isinstance(widget, ttk.Button) and widget not in (
            self.start_button,
            self.cancel_button,
        ):
            widget.configure(state=state)
        for child in widget.winfo_children():
            self._set_browse_state(child, state)

    def choose_recording(self) -> None:
        initial = None
        if self.recording is not None:
            initial = str(self.recording.parent)
        chosen = filedialog.askopenfilename(
            title="Choose a recording",
            initialdir=initial,
            filetypes=FILE_TYPES,
        )
        if not chosen:
            return
        path = Path(chosen)
        self.recording = path
        self._set_entry(self.recording_entry, str(path))
        self.file_name_var.set(f"File name: {path.name}")
        self.duration_var.set("Duration: reading…")
        self._set_status("Reading the recording…", "info")
        threading.Thread(
            target=self._probe_selected, args=(path,), daemon=True
        ).start()

    def _probe_selected(self, path: Path) -> None:
        try:
            info = probe_media(path)
        except Exception as exc:
            message = friendly_error(exc)

            def show_error() -> None:
                if self.running or self.recording != path:
                    return
                self.duration_var.set("Duration: —")
                self._set_status(message, "error")

            self._ui(show_error)
            return

        def show() -> None:
            if self.running or self.recording != path:
                return
            if info.duration is None:
                self.duration_var.set("Duration: unknown")
            else:
                self.duration_var.set(f"Duration: {format_clock(info.duration)}")
            if not info.has_audio:
                self._set_status("This file has no audio stream.", "error")
            else:
                self._set_status("Ready.", "info")

        self._ui(show)

    def choose_output_folder(self) -> None:
        initial = None
        if self.output_dir is not None:
            initial = str(self.output_dir)
        elif self.recording is not None:
            initial = str(self.recording.parent)
        chosen = filedialog.askdirectory(
            title="Choose an output folder",
            initialdir=initial,
        )
        if not chosen:
            return
        self.output_dir = Path(chosen)
        self._set_entry(self.output_entry, str(self.output_dir))

    def clear_output_folder(self) -> None:
        self.output_dir = None
        self._set_entry(self.output_entry, "")

    def on_start(self) -> None:
        if self.running:
            return
        if self.recording is None:
            self._set_status("Choose a recording first.", "error")
            return
        if not self.recording.is_file():
            self._set_status(
                "The recording could not be found. Choose the file again.",
                "error",
            )
            return
        if self.output_dir is not None and not self.output_dir.is_dir():
            self._set_status("The output folder could not be found.", "error")
            return
        model_name = self.model_box.get().strip()
        if model_name not in MODEL_SIZES:
            self._set_status("Choose a model size.", "error")
            return
        language_name = self.language_box.get().strip()
        if language_name not in self.language_by_name:
            self._set_status("Choose a language.", "error")
            return

        self.cancel_event.clear()
        self._set_running(True)
        self._set_progress(0)
        self._set_status("Starting…", "info")
        worker = threading.Thread(
            target=self._run_job,
            args=(
                self.recording,
                self.output_dir,
                bool(self.timestamps_var.get()),
                model_name,
                self.language_by_name[language_name],
            ),
            daemon=True,
        )
        worker.start()

    def _run_job(
        self,
        source: Path,
        output_dir: Path | None,
        timestamps: bool,
        model_name: str,
        language_code: str | None,
    ) -> None:
        try:
            txt_path, md_path, uncertain = transcribe_recording(
                source,
                output_dir,
                timestamps,
                model_name,
                language_code,
                cancel_event=self.cancel_event,
                on_status=lambda message: self._ui(
                    lambda text=message: self._set_status(text, "info")
                ),
                on_progress=lambda percent: self._ui(
                    lambda value=percent: self._set_progress(value)
                ),
                models=self.models,
                processes=self.processes,
            )
        except Cancelled:
            self._ui(self._finish_cancel)
        except Exception as exc:
            if not isinstance(
                exc, (MissingFileError, NoAudioError, ModelLoadError, MediaError)
            ):
                import traceback

                traceback.print_exc()
            self._ui(lambda error=exc: self._finish_error(error))
        else:
            self._ui(
                lambda txt=txt_path, md=md_path, unsure=uncertain: self._finish_ok(
                    txt, md, unsure
                )
            )

    def _finish_ok(self, txt_path: Path, md_path: Path, uncertain: bool) -> None:
        self._set_running(False)
        self._set_progress(100)
        message = f"Saved the transcript.\n{txt_path}\n{md_path}"
        if uncertain:
            message += (
                "\nThe detected language confidence is under 0.6. "
                "That is noted in the transcript."
            )
        self._set_status(message, "success")

    def _finish_cancel(self) -> None:
        self._set_running(False)
        self._set_progress(0)
        self._set_status("Cancelled. Nothing was saved.", "info")

    def _finish_error(self, exc: BaseException) -> None:
        self._set_running(False)
        self._set_progress(0)
        self._set_status(friendly_error(exc), "error")

    def on_cancel(self) -> None:
        if not self.running:
            return
        self.cancel_event.set()
        self.processes.stop()
        self._set_status("Cancelling… It will stop after the current step.", "info")

    def on_close(self) -> None:
        self.alive = False
        self.cancel_event.set()
        self.processes.stop()
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def main() -> None:
    root = tk.Tk()
    TranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
