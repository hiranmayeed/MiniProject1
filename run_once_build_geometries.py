# run_once_build_geometries.py
import json
import geopandas as gpd
import ee
from gee_gateway import initialize_gee

initialize_gee()

# Load your converted GeoJSON
gdf = gpd.read_file("data/external/telangana_districts.geojson")

# Check what the district name column is called
print("Columns:", gdf.columns.tolist())
print("Sample names:", gdf["DISTRICT_N"].head().tolist())

# Apply same name corrections as your pipeline
DISTRICT_NAME_CORRECTIONS = {
    "Jagtial"            : "Jagitial",
    "Jangoan"            : "Jangaon",
    "Kumuram Bheem"      : "Kumuram Bheem Asifabad",
    "Medchal-Malkajgiri" : "Medchal Malkajgiri",
    "Rangareddy"         : "Ranga Reddy",
    "Ranjanna Sircilla"  : "Rajanna Sircilla",
    "Warangal Rural"     : "Warangal (Rural)",
    "Warangal Urban"     : "Warangal (Urban)",
}

geometries = {}
skipped    = []

for _, row in gdf.iterrows():
    # Read district name — adjust "DISTRICT_N" if your column is named differently
    raw_name     = str(row["DISTRICT_N"]).strip().title()
    district_name = DISTRICT_NAME_CORRECTIONS.get(raw_name, raw_name)

    try:
        # Convert shapely polygon → GeoJSON dict → GEE geometry → dict
        # We store as dict because ee.Geometry objects aren't JSON-serialisable
        geom_geojson = row["geometry"].__geo_interface__
        ee_geom      = ee.Geometry(geom_geojson)
        geometries[district_name] = ee_geom.getInfo()
        print(f"  ✅ {district_name}")
    except Exception as e:
        print(f"  ❌ {district_name}: {e}")
        skipped.append(district_name)

# Save to disk
import os
os.makedirs("data/external", exist_ok=True)
with open("data/external/district_geometries.json", "w") as f:
    json.dump(geometries, f, indent=2)

print(f"\nSaved {len(geometries)} district geometries")
if skipped:
    print(f"Skipped: {skipped}")