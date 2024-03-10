import json
import os
from datetime import datetime as dt
from pathlib import Path
from typing import NamedTuple

from flask import Flask, render_template

from .mic import fork_recording_process
from .infer import fork_infer_process
from .settings import Settings


def _get_highest_n(d: dict[str, float], n: int) -> list[tuple[str, float]]:
    return [
        BirdResult(key, value)
        for key, value in sorted(
            d.items(), key=lambda k_v: k_v[1], reverse=True
        )[:n]
    ]


class BirdResult(NamedTuple):
    name: str
    score: float


class Result(NamedTuple):
    date: dt
    birds: list[BirdResult]

    def highest(self) -> BirdResult:
        return max(self.birds, key=lambda bird_result: bird_result.score)


def create_app(test_config=None):
    # create and configure the app

    app = Flask(__name__, instance_relative_config=True)

    settings = Settings(
        mic_pcms="plughw:CARD=Mic,DEV=0",
        audio_folder=Path(app.instance_path) / "audio",
        inference_folder=Path(app.instance_path) / "inference",
        model_path=Path("/home/pi/chorus-testing/reboot-03.jit"),
    )

    app.config.from_mapping(
        # SECRET_KEY="dev",
        # DATABASE=os.path.join(app.instance_path, "flaskr.sqlite"),
        **{key.upper(): val for key, val in settings.asdict().items()}
    )

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route("/")
    def root():
        results = [
            Result(
                dt.fromtimestamp(int(p.stem)),
                _get_highest_n(json.loads(p.read_text()), 5),
            )
            for p in sorted(
                app.config["INFERENCE_FOLDER"].glob("*.json"), reverse=True
            )
        ]
        # Filter out low scores
        results = [
            Result(
                result.date,
                [
                    bird_result
                    for bird_result in result.birds
                    if bird_result.score > 0.5
                ],
            )
            for result in results
            if result.highest().score > 0.5
        ]
        return render_template("root.html", results=results)

    fork_recording_process(settings)
    fork_infer_process(settings)

    return app
