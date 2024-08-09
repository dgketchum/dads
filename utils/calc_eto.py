from refet import Daily


def calc_asce_params(r, zw, lat, elev):
    asce = Daily(tmin=r['min_temp'],
                 tmax=r['max_temp'],
                 rs=r['rsds'],
                 ea=r['ea'],
                 uz=r['wind'],
                 zw=zw,
                 doy=r['doy'],
                 elev=elev,
                 lat=lat,
                 method='asce')

    vpd = asce.vpd[0]
    rn = asce.rn[0]
    u2 = asce.u2[0]
    tmean = asce.tmean[0]
    eto = asce.eto()[0]

    return tmean, vpd, rn, u2, eto


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
