from pathlib import Path
from typing import List

from pyproj import Transformer
import rasterio

from .data import load_range_map_meta


class Presence:
    """
    What are the odds that a bird is present in a location?
    """
    def __init__(self):
        self._df = load_range_map_meta().set_index('scientific_name')
        self._folder = Path(__file__).parents[1] / 'data' / 'ebird' / 'range'

    @property
    def known_scientific_names(self) -> List[str]:
        return sorted(self._df.index.tolist())

    def __call__(
        self,
        *,
        scientific_name: str,
        lat: float,
        lng: float,
        week: int,
        raise_on_unknown: bool=True
    ) -> float:
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
        scientific_name : str
            The scientific name of the bird. See the
            `known_scientific_names` property for a complete list.
            If your scientific name is not recognized, either an
            error is raised or -1 is returned (see raise_on_unknown argument).
        lat, lng : float
            Location to query. If your location is outside the dataset,
            then 0.0 is returned by default.
        week : int
            The week number from 1 to 52 inclusive.
        raise_on_unknown : bool
            Whether to raise an error or return -1 when the scientific_name
            is not in the range dataset. Default: raise an error

        Returns
        -------
        probability : float
            The probability that the bird can be observed at the given
            location at the given time of year. Might be -1 depending on
            the `raise_on_unknown` argument, but otherwise will be from
            0 to 1 inclusive.
        """
        try:
            row = self._df.loc[scientific_name]
        except KeyError:
            if raise_on_unknown:
                raise
            return -1.0
        filename = row['run_name'] + '_hr_2018_occurrence_median.tif'
        with rasterio.open(self._folder / filename) as raster:
            # lat/long -> raster coordinate system
            transform = Transformer.from_crs("EPSG:4326", raster.crs).transform
            try:
                return raster.read(week)[raster.index(*transform(lat, lng))]
            except IndexError:
                return 0.0
