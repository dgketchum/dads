# DADS Cube

*Full station-graph GNN pipeline for spatial meteorological prediction.*

The DADS Cube system decomposes target variables into station-derived climatological
backgrounds and daily anomalies predicted by a GNN on a neighbor graph. It requires
a complete data cube (stations.zarr, cube.zarr, graph.zarr), an autoencoder for station
embeddings, and a graph neural network for spatial interpolation.

!!! note "Status"
    Architecture and algorithm documentation is being migrated from internal notes.

## Chapters

- [System Architecture](architecture.md)
- [Data Provenance](data_provenance.md)
- [Data Cube Schema](cube_schema.md)
- [Algorithms](algorithms.md)
- [Design Decisions](decisions.md)
