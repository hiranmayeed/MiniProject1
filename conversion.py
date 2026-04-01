import os
import geopandas as gpd

os.environ["SHAPE_RESTORE_SHX"] = "YES"

gdf = gpd.read_file(r"C:\Users\hiran\RainfallAnalytics\map\TS_District_Boundary_33_FINAL.shp")

print(gdf.columns.tolist())
print(gdf.head(3))

# Reproject to WGS84 (lat/lon) — required for Folium web maps
gdf = gdf.to_crs(epsg=4326)

# Create the output folder if it doesn't exist
os.makedirs("data/external", exist_ok=True)

# Save as GeoJSON
gdf.to_file("data/external/telangana_districts.geojson", driver="GeoJSON")

print("Done. Districts saved:", gdf.shape[0])
print("District names sample:", gdf["DISTRICT_N"].tolist()[:5])