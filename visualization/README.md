# Heat map

Run from the repository root:

```sh
python3 -m venv visualization/.venv
visualization/.venv/bin/pip install -r visualization/requirements.txt
visualization/.venv/bin/python visualization/heatmap.py
```

This reads `data/geolife.parquet` and writes `output/geojson/heatmap.geojson.gz`.
