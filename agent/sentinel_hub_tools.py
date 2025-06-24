# geospatial_tools.py

import os
import json
import geopandas as gpd
import requests
import osmnx as ox
from dotenv import load_dotenv
from sentinelsat import geojson_to_wkt
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from pydantic import BaseModel

# Load environment variables
load_dotenv()
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

##########################
# Authentication Utility #
##########################
def get_access_token():
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)
    token = oauth.fetch_token(
        token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
        client_secret=client_secret,
        include_client_id=True
    )
    return token['access_token']


#####################
# Geo BBox Utilities #
#####################

class ExtractBox(BaseModel):
    location: str
    distance: float = 1000.0  # Default distance in meters
    filepath: str

def extract_bbox(**kwargs):
    input = ExtractBox(**kwargs)
    location = input.location
    distance = input.distance
    filepath = input.filepath

    # Generate GeoDataFrame from location
    gdf = ox.geocode_to_gdf(location)
    gdf.to_file(filepath, driver="GeoJSON")
    print(f"✅ GeoJSON file saved as {filepath}")

    # Compute bounding box
    minx, miny, maxx, maxy = gdf.total_bounds

    # ✅ Return structured output
    return {
        "file_path": filepath,
        "bbox": [minx, miny, maxx, maxy]
    }


class GetBox(BaseModel):
    file_path: str

def get_bbox(**kwargs):
    input = GetBox(**kwargs)
    file_path = input.file_path

    gdf = gpd.read_file(file_path)
    minx, miny, maxx, maxy = gdf.total_bounds

    # ✅ Return both bbox and path
    return {
        "file_path": file_path,
        "bbox": [minx, miny, maxx, maxy]
    }



##########################
# Evalscripts Dictionary #
##########################

evalscripts = {
    "dem": '''
function setup() {
  return { input: ["DEM"], output: { bands: 1 } };
}
function evaluatePixel(sample) {
  return [sample.DEM];
}''',
    "landcover": '''
function setup() {
  return { input: ["Map"], output: { bands: 1 } };
}
function evaluatePixel(sample) {
  return [sample.Map];
}''',
    "ndvi": '''
function setup() {
  return { input: ["B04", "B08"], output: { bands: 1 } };
}
function evaluatePixel(sample) {
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
  return [ndvi];
}''',
    "soil_saturation": '''
function setup() {
  return { input: ["VV", "VH"], output: { bands: 2 } };
}
function evaluatePixel(sample) {
  return [sample.VV, sample.VH];
}''',
    "ndwi": '''
function setup() {
  return { input: ["B03", "B08"], output: { bands: 1 } };
}
function evaluatePixel(sample) {
  let ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08);
  return [ndwi];
}''',
    "aod": '''
function setup() {
  return { input: ["AER_AI_340_380"], output: { bands: 1 } };
}
function evaluatePixel(sample) {
  return [sample.AER_AI_340_380];
}'''
}


##########################
# Dataset Type Mapping   #
##########################

SUPPORTED_DATASETS = {
    "dem": "DEM",
    "ndvi": "sentinel-2-l2a",
    "ndwi": "sentinel-2-l2a",
    "landcover": "sentinel-2-l2a",
    "soil_saturation": "sentinel-1-grd",
    "aod": "modis"
}


#####################
# Payload Generator #
#####################

class CreatePayload(BaseModel):
    dataset_type: str
    bbox: list
    time_from: str = "2023-01-01T00:00:00Z"
    time_to: str = "2023-12-31T23:59:59Z"
    evalscript: str = None

def create_payload(**kwargs):
    input = CreatePayload(**kwargs)

    dataset_type = input.dataset_type.lower()
    bbox = input.bbox
    time_from = input.time_from
    time_to = input.time_to
    evalscript = input.evalscript

    if dataset_type not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    if not evalscript:
        if dataset_type not in evalscripts:
            raise ValueError(f"No default evalscript available for dataset_type '{dataset_type}'")
        evalscript = evalscripts[dataset_type]

    print(f"🛠️ Creating payload for dataset: {dataset_type}, time: {time_from} to {time_to}")

    return {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{
                "type": SUPPORTED_DATASETS[dataset_type],
                "dataFilter": {
                    "timeRange": {
                        "from": time_from,
                        "to": time_to
                    }
                }
            }]
        },
        "output": {
            "width": 512,
            "height": 512,
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/tiff"}
            }]
        },
        "evalscript": evalscript
    }


##################
# Process Handler #
##################

class ProcessRequest(BaseModel):
    payload: dict
    filepath: str

def process_request(**kwargs) -> str:
    input = ProcessRequest(**kwargs)
    payload = input.payload
    filepath = input.filepath

    if not filepath.endswith(".tif"):
        filepath += ".tif"

    access_token = get_access_token()
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    print(f"🚀 Sending request to Sentinel Hub...")
    response = requests.post(
        "https://sh.dataspace.copernicus.eu/api/v1/process",
        headers=headers,
        data=json.dumps(payload)
    )

    if response.ok:
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"✅ Processed data saved to {filepath}")
        return filepath
    else:
        print("❌ Request failed:", response.status_code)
        try:
            print("📄 Error details:", response.json())
        except:
            print("📄 Error content:", response.text)
        raise RuntimeError(f"Sentinel Hub request failed with status {response.status_code}")
