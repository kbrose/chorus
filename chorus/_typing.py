from typing import Dict, List

from typing_extensions import TypedDict, Literal

XenoCantoRecording = TypedDict(
    'XenoCantoRecording',
    {
        'id': str,
        'gen': str,
        'sp': str,
        'ssp': str,
        'en': str,
        'rec': str,
        'cnt': str,
        'loc': str,
        'lat': str,
        'lng': str,
        'alt': str,
        'type': str,
        'url': str,
        'file': str,
        'file-name': str,
        'sono': Dict[Literal['small', 'med', 'large', 'full'], str],
        'lic': str,
        'q': Literal['A', 'B', 'C', 'D', 'E'],
        'length': str,
        'time': str,
        'date': str,
        'uploaded': str,
        'also': List[str],
        'rmk': str,
        'bird-seen': Literal['yes', 'no'],
        'playback-used': Literal['yes', 'no']
    }
)

XenoCantoResponse = TypedDict(
    'XenoCantoResponse',
    {
        'numRecordings': str,
        'numSpecies': str,
        'page': int,
        'numPages': int,
        'recordings': List[XenoCantoRecording]
    }
)
