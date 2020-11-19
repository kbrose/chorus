from pathlib import Path
from typing import Dict
import json

from pyproj import Transformer
import affine
import numpy as np


class Presence:
    """
    What are the odds that a bird is present in a location?
    """
    def __init__(self):
        self._folder = Path(__file__).parents[1] / 'data' / 'ebird' / 'range'

        with open(self._folder / 'meta.json') as f:
            meta = json.load(f)
        self.scientific_names = meta['scientific_names']

        # lat/long -> coordinate system of raster
        transformer = Transformer.from_crs("EPSG:4326", meta['crs'])
        self._crs_transform = transformer.transform
        # coordinate system of raster -> array indices
        self._raster_transform = ~affine.Affine(*meta['transform'])

    def __call__(
        self,
        *,
        lat: float,
        lng: float,
        week: int,
    ) -> Dict[str, float]:
        """
        Return the probability that a bird is present.

        The EBird folks define this probability as follows:

            This [value] represents the expected probability of occurrence
            of the species, ranging from 0 to 1, on an eBird Traveling Count
            by a skilled eBirder starting at the optimal time of day with
            the optimal search duration and distance that maximizes detection
            of that species in a region.

        https://cornelllabofornithology.github.io/ebirdst/articles/ebirdst-introduction.html#occurrence_median

        Inputs
        ------
        lat, lng : float
            Location to query. If your location is outside the dataset,
            then 0.0 is returned by default.
        week : int
            The week number from 1 to 52 inclusive.

        Returns
        -------
        probabilities : Dict[str, float]
            A map of scientific name -> probability that the bird can be
            observed at the given location at the given time of year.
            See above for rigorous definition of the probability.
        """
        if not 1 <= week <= 52:
            raise ValueError('week must be between 1 and 52 inclusive')

        # "load" data from disk (use mmap to do lazy loading of range data)
        data = np.load(self._folder / 'ranges.npy', mmap_mode='r')

        try:
            col, row = self._raster_transform * self._crs_transform(lat, lng)
            values = data[:, int(week), int(row), int(col)].clip(0, 1)
            values = np.nan_to_num(values, 0.0)
        except IndexError:
            values = np.zeros_like(self.scientific_names, dtype='float32')
        return dict(zip(self.scientific_names, values))
