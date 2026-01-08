# Data Directory

Reference datasets for sim-city simulations.

## Expected Files

This directory should contain:

- `GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg` - GHSL Functional Urban Areas
  - Download from: https://ghsl.jrc.ec.europa.eu/download.php

## Data Sources

### POIs
- **Overture Places**: Global POI data, fetched via DuckDB
- **OpenStreetMap**: Global POI data, fetched via osmnx

### Population
- **US Census ACS**: US tract/block group demographics
- **GHS-POP**: Global 100m population grid (coming soon)

### Boundaries
- **US Census TIGER**: US geographic boundaries
- **GHSL FUA**: Global functional urban areas

## Notes

Large data files should not be committed to the repository.
Add them to `.gitignore` as needed.
