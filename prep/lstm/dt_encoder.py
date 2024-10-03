import numpy as np
import pandas as pd


def datetime_encoded(df, units):
    mapping = {un: pd.to_timedelta('1' + un).value
               for un in ['day', 'hour', 'minute', 'second',
                          'millisecond', 'microsecond', 'nanosecond']}
    mapping['week'] = pd.to_timedelta('1W').value
    mapping['year'] = 365.2425 * 24 * 60 * 60 * 10 ** 9
    index_nano = df.index.view(np.int64)
    datetime = dict()
    for unit in units:
        if unit not in mapping:
            raise ValueError()
        nano_sec = index_nano * (2 * np.pi / mapping[unit])
        datetime[unit + '_sin'] = np.sin(nano_sec)
        datetime[unit + '_cos'] = np.cos(nano_sec)
    return pd.DataFrame(datetime, index=df.index, dtype=np.float32)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
