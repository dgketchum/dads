import os
import json
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from tsl.datasets.prototypes.datetime_dataset import DatetimeDataset
from tsl.datasets.prototypes.mixin import MissingValuesMixin
from tsl.datasets.prototypes.mixin import TemporalFeaturesMixin
from tsl.utils.parser_utils import ArgParser
from tsl.datasets.prototypes import PandasDataset
from tsl.ops.similarities import gaussian_kernel


class Meteorology(PandasDataset):
    similarity_options = {'distance'}

    def __init__(self,
                 points,
                 target_parameter,
                 root,
                 data_subdir='compiled_csv',
                 freq='D',
                 na_threshold=0.5,
                 knn=5,
                 ):

        self.root = root
        self.data_subdir = data_subdir
        self.target_parameter = target_parameter
        self.points = pd.read_csv(points, index_col='fid')

        self.na_threshold = na_threshold

        self.knn = knn

        df, validation_dct, distances, mask = self.load()
        super().__init__(target=df,
                         mask=mask,
                         freq=freq,
                         similarity_score='distance',
                         temporal_aggregation='mean',
                         spatial_aggregation='mean',
                         default_splitting_method='val',
                         name=target_parameter)

        self.add_covariate('dist', distances, pattern='n n')
        # self.set_eval_mask(validation_dct)

    def load_raw(self):
        """Simplistic input data of only target values"""

        evaluation = os.path.join(self.root_dir, 'evaluation.json')
        timeseries = os.path.join(self.root_dir, 'timeseries.csv')
        if all([os.path.exists(evaluation), os.path.exists(timeseries)]):
            df = pd.read_csv(timeseries, index_col=0, parse_dates=True)
            self.points = self.points.loc[df.columns]
            with open(evaluation, 'r') as fp:
                eval_ = json.load(fp)

            return df, eval_

        csv_dir = os.path.join(self.root, self.data_subdir)
        first, df, eval_ = True, None, {}
        for i, r in self.points.iterrows():

            # validation: 1, training: 0
            eval_[i] = int(r['val'])

            station_data_file = os.path.join(csv_dir, '{}.csv'.format(i))

            if not os.path.exists(station_data_file):
                continue

            station_data = pd.read_csv(station_data_file, index_col=0, parse_dates=True)
            s = station_data[f'{self.target_parameter}_obs']
            s.name = i
            if first:
                df = pd.DataFrame(s)
                first = False
            else:
                df = pd.concat([df, s], axis=1, ignore_index=False)

        drop_cols = [c for c in df.columns if df[c].isna().sum(axis=0) / df.shape[0] > self.na_threshold]
        df.drop(columns=drop_cols, inplace=True)
        self.points = self.points.loc[df.columns]
        df = df.fillna(-1.0)

        df.to_csv(timeseries)

        eval_ = {k: v for k, v in eval_.items() if k not in drop_cols}
        with open(evaluation, 'w') as fp:
            json.dump(eval_, fp, indent=4)

        return df, eval_

    def load(self, impute_zeros=True):
        df, _eval = self.load_raw()
        distances_ = self.get_distances()
        mask_ = (df.values != -1.0).astype('uint8')
        if impute_zeros:
            df = df.replace(to_replace=0., method='ffill')
        return df, _eval, distances_, mask_

    def get_distances(self):
        dist_file = os.path.join(self.root_dir, 'distances.npy')
        if not os.path.exists(dist_file):
            dist = self._calculate_haversine_distances(dist_file)
        else:
            dist = np.load(dist_file)
        return dist

    def generate_edge_index(self, distances):

        # TODO: implement high dimensional similarity here

        k_nearest_indices = np.argsort(distances, axis=1)[:, 1:self.k_nearest + 1]
        row_indices = np.repeat(np.arange(self.points, self.knn))
        spatial_edges = np.column_stack([row_indices, k_nearest_indices.ravel()])

        features = self.points[['ELEV', 'longitude', 'latitude']].values

        scaler = MinMaxScaler()
        normalized_features = np.column_stack([
            scaler.fit_transform(features[:, i].reshape(-1, 1)) for i in range(features.shape[1])
        ])
        similarity_matrix = cosine_similarity(normalized_features)
        k_nearest_feature_indices = np.argsort(-similarity_matrix, axis=1)[:, 1:self.k_nearest + 1]
        row_indices = np.repeat(np.arange(len(self.points)), self.k_nearest)
        feature_edges = np.column_stack([row_indices, k_nearest_feature_indices.ravel()])
        all_edges = np.unique(np.vstack([spatial_edges, feature_edges]), axis=0)
        combined_edges = all_edges[all_edges[:, 0] != all_edges[:, 1]]

        G = nx.Graph()
        G.add_edges_from(combined_edges)
        if not nx.is_connected(G):
            for component in list(nx.connected_components(G)):
                main_component = list(nx.connected_components(G))[0]
                if component != main_component:
                    node_from_main = list(main_component)[0]
                    node_from_component = list(component)[0]
                    G.add_edge(node_from_main, node_from_component)

        combined_edges = np.array(G.edges)
        index_to_staid = {i: staid for i, staid in enumerate(self.points[self.index_col])}
        edge_lines = []
        to_, from_ = [], []
        for i, j in combined_edges:
            point1 = Point(self.points.iloc[i].geometry)
            point2 = Point(self.points.iloc[j].geometry)
            line = LineString([point1, point2])
            from_.append(self.points.iloc[i][self.index_col])
            to_.append(self.points.iloc[j][self.index_col])
            edge_lines.append(line)

        gdf_edges = gpd.GeoDataFrame({'geometry': edge_lines})
        gdf_edges['to'] = to_
        gdf_edges['from'] = from_

        with open(os.path.join(self.output_dir, 'edge_indx_map.json'), 'w') as f:
            json.dump(index_to_staid, f)

        np.savetxt(os.path.join(self.output_dir, 'edge_indx.np'), combined_edges)

        gdf_edges.to_file(os.path.join(self.output_dir, 'edges.shp'), epsg='EPSG:4326', engine='fiona')

        return combined_edges, index_to_staid

    def _calculate_haversine_distances(self, out_file):

        coordinates = self.points[['lon', 'lat']].values
        dst = []
        for i in range(len(coordinates)):
            dst.append([
                haversine_distance(coordinates[i][0], coordinates[i][1],
                                   coordinates[j][0], coordinates[j][1])
                for j in range(len(coordinates))
            ])
        dst = np.array(dst)

        for i in range(dst.shape[0]):
            kth_smallest = np.partition(dst[i], self.knn - 1)[self.knn - 1]
            dst[i][dst[i] > kth_smallest] = np.inf

        np.save(out_file, dst)
        return dst

    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            finite_dist = self.dist.reshape(-1)
            finite_dist = finite_dist[~np.isinf(finite_dist)]
            sigma = finite_dist.std()
            return gaussian_kernel(self.dist, sigma)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--root', type=str, default='./data')
        parser.add_argument('--points', type=str, default='points.csv')
        parser.add_argument('--csv_dir', type=str, default='./csv_data')
        return parser


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + \
        np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * \
        np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


if __name__ == '__main__':
    pass

# ========================= EOF ====================================================================
