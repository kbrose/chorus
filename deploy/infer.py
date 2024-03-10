import json
import multiprocessing as mp
from pathlib import Path
import time
from collections import deque
from datetime import datetime as dt, timedelta as td

import librosa
import torch
from torch.nn.functional import sigmoid

from .settings import Settings


def _load_model_and_classes(
    settings: Settings,
) -> tuple[torch.jit.ScriptModule, list[str]]:
    extra_files = {"targets.json": ""}
    model = torch.jit.load(
        str(settings.model_path.expanduser().absolute()),
        _extra_files=extra_files,
    )
    classes = json.loads(extra_files["targets.json"])
    return model, classes


def _infer_on_file(
    model: torch.jit.ScriptModule,
    classes: list[str],
    filepath: Path,
    settings: Settings,
) -> dict[str, float]:
    x = torch.tensor(
        librosa.load(filepath, sr=settings.sample_rate)[0][None, :]
    )
    return dict(
        zip(
            classes,
            [float(v) for v in sigmoid(model(x)[0][0]).numpy()],
        )
    )


def _infer_loop(
    model: torch.jit.ScriptModule, classes: list[str], settings: Settings
):
    settings.inference_folder.mkdir(parents=True, exist_ok=True)
    check_every = max(settings.audio_file_seconds // 3, 2)
    recently_completed: deque[Path] = deque(maxlen=10)
    now = dt.now()
    while True:
        prev = now
        now = dt.now()
        # Check for new files between previous for loop time and now, but
        # go back an additional 5 seconds just in case arecord was slow
        # getting the file out
        for delta in range(0, int((now - prev).total_seconds()) + 5):
            t = now - td(seconds=delta)
            file = settings.audio_folder / t.strftime(
                settings.audio_file_name_pattern
            )
            if file.exists() and file not in recently_completed:
                recently_completed.append(file)
                results = _infer_on_file(model, classes, file, settings)
                # TODO: Use geo presence modifier
                with open(
                    settings.inference_folder / file.with_suffix(".json").name,
                    "w",
                ) as f:
                    # TODO: Put this in database instead
                    json.dump(results, f)

        sleep_time = max(0, check_every - (dt.now() - now).total_seconds())
        time.sleep(sleep_time)


def fork_infer_process(settings: Settings) -> mp.Process:
    model, classes = _load_model_and_classes(settings)
    process = mp.Process(target=_infer_loop, args=(model, classes, settings))
    process.start()
    return process
