import earthaccess


def get_nldas(dst):
    auth = earthaccess.login()
    print('earthdata access authenticated')
    results = earthaccess.search_data(
        doi='10.5067/THUF4J1RLSYG',
        temporal=('1990-01-01', '2023-12-31'))
    files = earthaccess.download(results, dst)


if __name__ == '__main__':
    folder = '/data/ssd1/nldas2'

    get_nldas(folder)
# ========================= EOF ====================================================================
