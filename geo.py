#!/usr/bin/python3.8
# coding=utf-8
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as ctx
import sklearn.cluster
import seaborn as sns
import numpy as np
import os


# muzeze pridat vlastni knihovny


def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """ Konvertovani dataframe do geopandas.GeoDataFrame se spravnym kodovanim"""

    df = df.dropna(subset=['d', 'e'])
    # puvodni data v Krovakove zobrazeni
    gdf = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df['d'], df['e']), crs='EPSG:5514'
    )

    # pro zabraneni zkresleni prevedeme do Web Mercator projection
    gdf = gdf.to_crs('EPSG:3857')

    return gdf


def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """ Vykresleni grafu se dvema podgrafy podle lokality nehody """

    # priprava dat
    msk = gdf[gdf['region'] == 'MSK']
    in_town = msk[msk['p5a'] == 1]
    out_of_town = msk[msk['p5a'] == 2]

    # graf jako celek
    figure = plt.figure(figsize=(16, 8))
    figure.suptitle('Nehody v Moravskoslezském kraji')
    gridspec = figure.add_gridspec(1, 2)

    # podgraf obec
    ax = figure.add_subplot(gridspec[0, 0])
    ax.axes.set_title('V obci')
    in_town_plot = in_town.plot(ax=ax, alpha=0.3, color="tab:red")
    ctx.add_basemap(
        ax,
        crs=gdf.crs.to_string(),
        source=ctx.providers.Stamen.TonerLite,
    )
    ax.axis("off")

    # podgraf mimo obec
    ax = figure.add_subplot(gridspec[0, 1])
    ax.axes.set_title('Mimo obec')
    out_of_town_plot = out_of_town.plot(ax=ax, alpha=0.3, color="tab:blue")
    ax.axis("off")
    ctx.add_basemap(
        ax,
        crs=gdf.crs.to_string(),
        source=ctx.providers.Stamen.TonerLite,
    )
    figure.tight_layout()

    # vystupy grafu
    if fig_location:
        d = os.path.dirname(fig_location)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        figure.savefig(fig_location)

    if show_figure:
        figure.show()


def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """ Vykresleni grafu s lokalitou vsech nehod v kraji shlukovanych do clusteru """

    # uprava dat
    msk = gdf[gdf['region'] == 'MSK']
    msk_orig = msk.copy()
    coords = np.dstack([msk.d, msk.e]).reshape(-1, 2)
    # 13
    db = sklearn.cluster.MiniBatchKMeans(n_clusters=25).fit(coords)
    # 14
    msk['cluster'] = db.labels_
    msk['cluster2'] = db.labels_

    # 15
    msk_d = msk.dissolve(by='cluster', aggfunc={'cluster2': 'count'}).rename(columns={'cluster2': 'count'})
    # 18
    gdf_coords = geopandas.GeoDataFrame(geometry=geopandas.points_from_xy(db.cluster_centers_[:, 0], db.cluster_centers_[:, 1]))
    # 19
    msk_m = msk_d.merge(gdf_coords, left_on='cluster', right_index=True)
    msk_g = msk_m.set_geometry('geometry_y')
    msk_g = msk_g.drop(columns='geometry_x')
    msk_g = msk_g.set_crs(epsg=5514)
    msk_g = msk_g.to_crs(epsg=3857)

    # graf
    figure = plt.figure(figsize=(10, 10))
    figure.suptitle('Nehody v Moravskoslezském kraji')
    ax = figure.add_subplot()
    ax.axis("off")

    msk_orig.plot(ax=ax, alpha=0.1, facecolor='grey', markersize=4)
    msk_g.plot(ax=ax, column='count', markersize=(msk_g['count'] / 5), legend=True, alpha=0.6)
    ctx.add_basemap(
        ax,
        # crs=msk_orig.crs.to_string(),
        crs='EPSG:3857',
        source=ctx.providers.Stamen.TonerLite,
    )
    figure.tight_layout()
    figure.show()

    # vystupy grafu
    if fig_location:
        d = os.path.dirname(fig_location)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        figure.savefig(fig_location)

    if show_figure:
        figure.show()


if __name__ == '__main__':
    # zde muzete delat libovolne modifikace
    df = pd.read_pickle('accidents.pkl.gz')
    gdf = make_geo(df)
    plot_geo(gdf, 'geo1.png', True)
    plot_cluster(gdf, "geo2.png", True)

    fig_location = None
    show_figure = True
