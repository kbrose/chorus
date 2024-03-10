import multiprocessing as mp
import subprocess as sp

from .settings import Settings


def _record_indefinitely(settings: Settings):
    """Start running arecord indefinitely."""
    # arecord -D plughw:CARD=Mic,DEV=0 -f cd --max-file-time 5 --use-strftime "%s.wav"
    args = [
        "arecord",
        "-D",
        settings.mic_pcms,
        "-f",
        "S16_LE",
        "-r",
        str(settings.sample_rate),
        "--max-file-time",
        str(settings.audio_file_seconds),
        "--use-strftime",
        str(
            settings.audio_folder.expanduser().absolute()
            / settings.audio_file_name_pattern
        ),
    ]
    return sp.run(args)


def fork_recording_process(settings: Settings) -> mp.Process:
    """Start running arecord in a separate process."""
    process = mp.Process(target=_record_indefinitely, args=(settings,))
    process.start()
    return process
