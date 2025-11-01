import os
import rasterio
import numpy as np

def test_population_at_lat_lon():
    """Extract population value at latitude 10 and longitude 20 from LandScan 2010 GeoTIFF."""
    tif_path = "population_data/landscan-global-2012.tif"
    assert os.path.exists(tif_path), f"GeoTIFF file not found at {tif_path}"

    with rasterio.open(tif_path) as src:
        lon, lat = 20, 10
        row, col = src.index(lon, lat)
        assert 0 <= row < src.height, "Row out of bounds"
        assert 0 <= col < src.width, "Column out of bounds"

        # Read a single pixel only
        population = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]

        assert population >= 0, "Population value is negative"
        assert not np.isnan(population), "Population value is NaN"

        print(f"Population at lat={lat}, lon={lon}: {population}")

if __name__ == "__main__":
    test_population_at_lat_lon()
