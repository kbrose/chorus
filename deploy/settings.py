from pathlib import Path
from dataclasses import asdict, dataclass, field


class MisConfiguredError(Exception):
    pass


@dataclass
class Settings:
    # Settable settings
    mic_pcms: str
    audio_folder: Path
    inference_folder: Path
    model_path: Path

    # Settings that aren't settable. Basically, a nice place for global state.
    sample_rate: int = field(default=30_000, init=False)
    audio_file_seconds: int = field(default=15, init=False)
    audio_file_name_pattern: str = field(default=r"%s.wav", init=False)

    def __post_init__(self):
        if self.audio_file_seconds < 5:
            raise MisConfiguredError("audio_file_seconds must be >= 5")

    def asdict(self):
        return asdict(self)
