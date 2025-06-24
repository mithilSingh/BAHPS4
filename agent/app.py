from typing import List
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import mapping
from pydantic import BaseModel, ConfigDict, Field
import numpy as np
import json
import geopandas as gpd
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from tqdm import tqdm
import whitebox

import os
from whitebox import WhiteboxTools

import rasterio
from rasterio.enums import Compression

### VECTOR TOOLS ###

class LoadVector(BaseModel):
    path: str

def load_vector(**kwargs) -> str:
    input = LoadVector(**kwargs)
    return input.path


class ReprojectVector(BaseModel):
    path: str
    epsg: str

def reproject_vector(**kwargs) -> dict:
    input = ReprojectVector(**kwargs)
    gdf = gpd.read_file(input.path)
    gdf = gdf.to_crs(epsg=input.epsg)
    output_path = "reprojected.geojson"
    gdf.to_file(output_path, driver="GeoJSON")
    return {"path": output_path}


class BufferShape(BaseModel):
    path: str
    distance: int

def buffer_shape(**kwargs) -> dict:
    input = BufferShape(**kwargs)
    gdf = gpd.read_file(input.path)
    projected = gdf.to_crs(epsg=32644)
    projected["geometry"] = projected.geometry.buffer(input.distance)
    buffered = projected.to_crs(gdf.crs)
    output_path = "buffered.geojson"
    buffered.to_file(output_path, driver="GeoJSON")
    return {"path": output_path}


class UnionShape(BaseModel):
    path1: str
    path2: str

def union_shapes(**kwargs) -> dict:
    input = UnionShape(**kwargs)
    gdf1 = gpd.read_file(input.path1)
    gdf2 = gpd.read_file(input.path2)
    result = gpd.overlay(gdf1, gdf2, how="union")
    output_path = "union.geojson"
    result.to_file(output_path, driver="GeoJSON")
    return {"path": output_path}

def intersect_shapes(**kwargs) -> dict:
    input = UnionShape(**kwargs)
    gdf1 = gpd.read_file(input.path1)
    gdf2 = gpd.read_file(input.path2)
    result = gpd.overlay(gdf1, gdf2, how="intersection")
    output_path = "intersection.geojson"
    result.to_file(output_path, driver="GeoJSON")
    return {"path": output_path}

def distance_between_shapes(**kwargs) -> float:
    input = UnionShape(**kwargs)
    gdf1 = gpd.read_file(input.path1)
    gdf2 = gpd.read_file(input.path2)
    return gdf1.geometry.distance(gdf2.geometry.iloc[0]).min()


### RASTER TOOLS ###

class ClipRaster(BaseModel):
    raster_path: str
    vector_path: str
    output_path: str

def clip_raster(**kwargs) -> dict:
    input = ClipRaster(**kwargs)
    with rasterio.open(input.raster_path) as src:
        gdf = gpd.read_file(input.vector_path)
        shapes = [mapping(geom) for geom in gdf.geometry]
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })
    with rasterio.open(input.output_path, "w", **out_meta) as dest:
        dest.write(out_image)
    return {"path": input.output_path}

import os
class CalculateSlope(BaseModel):
    dem_path: str
    output_path: str

import os
from whitebox import WhiteboxTools

import rasterio
from rasterio.enums import Compression

def convert_dem_to_supported_compression(input_path) -> str:
    output_path = input_path.replace(".tif", "_deflate.tif")
    with rasterio.open(input_path) as src:
        profile = src.profile
        profile.update({
            "compress": "DEFLATE",  # ✅ compatible compression
            "driver": "GTiff"
        })
        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                dst.write(src.read(i), i)
    return output_path


# Convert input DEM

def calculate_slope(**kwargs) -> dict:
    input = CalculateSlope(**kwargs)
    

    dem_path = os.path.abspath(input.dem_path)
    output_path = os.path.abspath(input.output_path)
    dem_path = convert_dem_to_supported_compression(dem_path)

    print("📍 DEM Path:", dem_path)
    print("📍 Output Path:", output_path)

    if not os.path.exists(dem_path):
        raise FileNotFoundError(f"❌ DEM file not found at {dem_path}")

    wbt = WhiteboxTools()
    exe_dir = r"E:/CodeBook/ISRO/.venv/Lib/site-packages/whitebox"
    wbt.set_whitebox_dir(exe_dir)

    # Enable verbose logging
    wbt.verbose = True

    # Print command before running
    cmd = f"slope --dem='{dem_path}' --output='{output_path}'"
    print(f"🧪 Running WhiteboxTools command: {cmd}")

    # Run the slope tool
    try:
        wbt.slope(dem=dem_path, output=output_path)
    except Exception as e:
        print("❌ Exception during slope calculation:", e)
        raise

    # Check for output
    if not os.path.exists(output_path):
        print("⚠️ Output file not found after execution!")
        raise RuntimeError("⚠️ Slope output was not created at expected location.")

    print("✅ Slope file created successfully.")
    return {"path": output_path}



def read_raster_array(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


class CalculateNDVI(BaseModel):
    nir_path: str
    red_path: str

def calculate_ndvi(**kwargs) -> dict:
    input = CalculateNDVI(**kwargs)
    nir = read_raster_array(input.nir_path)
    red = read_raster_array(input.red_path)
    ndvi = (nir - red) / (nir + red + 1e-6)
    output_path = "ndvi.tif"
    with rasterio.open(input.nir_path) as src:
        meta = src.meta.copy()
    meta.update(count=1)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(ndvi.astype(rasterio.float32), 1)
    return {"path": output_path}


class ClassifyFlood(BaseModel):
    ndvi_path: str
    dem_path: str
    ndvi_thresh: float = 0.2
    elevation_thresh: float = 250

def classify_flood_risk(**kwargs) -> dict:
    input = ClassifyFlood(**kwargs)
    ndvi = read_raster_array(input.ndvi_path)
    dem = read_raster_array(input.dem_path)
    mask = np.logical_and(ndvi < input.ndvi_thresh, dem < input.elevation_thresh)
    output_path = "flood_map.geojson"

    # Convert mask to polygons
    from skimage import measure
    from shapely.geometry import Polygon
    shapes = measure.find_contours(mask.astype(np.uint8), 0.5)
    polygons = [Polygon(shape[:, [1, 0]]) for shape in shapes if len(shape) > 2]
    gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
    gdf.to_file(output_path, driver="GeoJSON")
    return {"path": output_path}


class RasterStats(BaseModel):
    raster_path: str

def raster_stats(**kwargs) -> dict:
    input = RasterStats(**kwargs)
    data = read_raster_array(input.raster_path)
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "sum": float(np.sum(data)),
        "path": input.raster_path
    }


### GEOJSON HELPERS ###

class ExtractBbox(BaseModel):
    path: str

def extract_bbox_from_geojson(**kwargs) -> dict:
    input = ExtractBbox(**kwargs)
    gdf = gpd.read_file(input.path)
    bounds = list(gdf.total_bounds)
    return {
        "bbox": bounds,
        "path": input.path
    }


class SaveGeojson(BaseModel):
    geojson: str
    output_path: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

def save_geojson(**kwargs) -> dict:
    input = SaveGeojson(**kwargs)
    gdf = gpd.read_file(json.loads(input.geojson))
    gdf.to_file(input.output_path, driver="GeoJSON")
    return {"path": input.output_path}


class AnalyzeFloodZones(BaseModel):
    geojson_path: str
    sample_size: int = Field(default=20, ge=1, description="Number of sampled flood zones for analysis")

def analyze_flood_zones(**kwargs) -> dict:
    input = AnalyzeFloodZones(**kwargs)
    geojson_path = input.geojson_path
    sample_size = input.sample_size
    # Load the flood zones
    gdf = gpd.read_file(geojson_path)

    if gdf.empty:
        return {"message": "⚠️ No flood risk zones found in the GeoJSON."}

    # Calculate extent
    bbox = list(gdf.total_bounds)
    total_zones = len(gdf)

    # Get centroid points for sampling
    sample_points = gdf.centroid.sample(n=min(sample_size, total_zones), random_state=42)
    
    # Setup reverse geocoder
    geolocator = Nominatim(user_agent="flood_analysis")

    tqdm.pandas(desc="🔍 Reverse geocoding flood zones...")
    locations = sample_points.progress_apply(
        lambda point: geolocator.reverse((point.y, point.x), exactly_one=True, timeout=10).address
        if point.is_valid else "Unknown"
    )

    return {
        "total_flood_zones": total_zones,
        "bounding_box": bbox,
        "sampled_affected_areas": list(locations.dropna().unique())
    }
