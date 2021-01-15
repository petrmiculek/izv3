import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import contextily as ctx
import sklearn.cluster
import numpy as np
import os
from collections import defaultdict
import seaborn as sns
import locale

locale.setlocale(locale.LC_ALL, locale='cs_CZ.utf8')
# check locale availability in shell: $ locale -a
# used to set comma as the decimal separator

_brands_number_to_name = {
    # 0: 'neznama',
    1: 'ALFA-ROMEO',
    2: 'AUDI',
    3: 'AVIA',
    4: 'BMW',
    5: 'CHEVROLET',
    6: 'CHRYSLER',
    7: 'CITROEN',
    8: 'DACIA',
    9: 'DAEWOO',
    10: 'DAF',
    11: 'DODGE',
    12: 'FIAT ',
    13: 'FORD',
    14: 'GAZ-VOLHA',
    15: 'FERRARI',
    16: 'HONDA',
    17: 'HYUNDAI',
    18: 'IFA',
    19: 'IVECO',
    20: 'JAGUAR',
    21: 'JEEP',
    22: 'LANCIA',
    23: 'LAND ROVER',
    24: 'LIAZ',
    25: 'MAZDA',
    26: 'MERCEDES',
    27: 'MITSUBISHI',
    28: 'MOSKVIČ',
    29: 'NISSAN',
    30: 'OLTCIT',
    31: 'OPEL',
    32: 'PEUGEOT',
    33: 'PORSCHE',
    34: 'PRAGA',
    35: 'RENAULT',
    36: 'ROVER',
    37: 'SAAB',
    38: 'SEAT',
    39: 'ŠKODA',
    40: 'SCANIA',
    41: 'SUBARU',
    42: 'SUZUKI',
    43: 'TATRA',
    44: 'TOYOTA',
    45: 'TRABANT',
    46: 'VAZ',
    47: 'VOLKSWAGEN',
    48: 'VOLVO',
    49: 'WARTBURG',
    50: 'ZASTAVA',
    51: 'AGM',
    52: 'ARO',
    53: 'AUSTIN',
    54: 'BARKAS',
    55: 'DAIHATSU',
    56: 'DATSUN',
    57: 'DESTACAR',
    58: 'ISUZU',
    59: 'KAROSA',
    60: 'KIA',
    61: 'LUBLIN',
    62: 'MAN',
    63: 'MASERATI',
    64: 'MULTICAR',
    65: 'PONTIAC',
    66: 'ROSS',
    67: 'SIMCA',
    68: 'SSANGYONG',
    69: 'TALBOT',
    70: 'TAZ',
    71: 'ZAZ',
    72: 'BOVA',
    73: 'IKARUS',
    74: 'NEOPLAN',
    75: 'OASA',
    76: 'RAF',
    77: 'SETRA',
    78: 'SOR',
    79: 'APRILIA',
    80: 'CAGIVA',
    81: 'ČZ',
    82: 'DERBI',
    83: 'DUCATI',
    84: 'GILERA',
    85: 'HARLEY',
    86: 'HERO',
    87: 'HUSQVARNA',
    88: 'JAWA',
    89: 'KAWASAKI',
    90: 'KTM',
    91: 'MALAGUTI',
    92: 'MANET',
    93: 'MZ',
    94: 'PIAGGIO',
    95: 'SIMSON',
    96: 'VELOREX',
    97: 'YAMAHA',
    98: 'jiné vyrobené v ČR',
    99: 'jiné vyrobené mimo ČR',
}
brands_number_to_name = defaultdict(lambda: 'neznama', _brands_number_to_name)
max_age = 30


def plot_age_total(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    global max_age
    x, y = np.unique(df['stari'], return_counts=True)

    # data stačí v tisících
    # noinspection PyAugmentAssignment
    y = y / 1000

    fig = plt.figure(figsize=(6, 4))
    ax = sns.lineplot(x=x[:max_age + 5], y=y[:max_age + 5], color='#bc5090', marker='o')
    ax.set_title('Celkové zastoupení stáří havarovaných vozidel')
    ax.set_ylabel('počet vozidel [tisíc]')
    ax.set_xlabel('stáří vozidla [let]')
    ax.set_xlim(left=0, right=max_age)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.subplots_adjust(left=0.15)

    if fig_location:
        d = os.path.dirname(fig_location)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        fig.savefig(fig_location)

    if show_figure:
        fig.show()


def plot_age_yearly(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    global max_age

    fig = plt.figure(figsize=(6, 4))

    palette = sns.color_palette(['#addd8e', '#78c679', '#41ab5d', '#238443', '#005a32'])
    ax = sns.kdeplot(data=df[['stari', 'rok_nehody']], x='stari',
                     hue='rok_nehody', palette=palette,
                     common_norm=False)  # každý rok je normalizován zvlášť

    ax.get_legend().set_title('rok nehody')
    ax.set_xlim(left=0, right=max_age)
    ax.set_ylim(bottom=0)
    ax.set_title('Zastoupení stáří havarovaných vozidel dle roků')
    ax.yaxis.set_major_formatter('{x:#.1n}%')  # n-formátování respektuje Locale nastavení
    ax.set_xlabel('stáří vozidla [let]')
    ax.set_ylabel('podíl na celkovém počtu nehod [%]')
    fig.tight_layout()
    fig.show()
    fig.subplots_adjust(left=0.15)

    if fig_location:
        d = os.path.dirname(fig_location)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        fig.savefig(fig_location)

    if show_figure:
        fig.show()


def plot_brands_total(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    count_total = len(df.index)
    brands = df.groupby(['znacka']) \
        .aggregate({'znacka': 'count'}) \
        .rename(columns={'znacka': 'pocet'}) \
        .sort_values(by='pocet', ascending=False)

    for i in [0, 98, 99]:
        # odstraneni zaznamu neuvedenych znacek
        idx = brands_number_to_name[i]
        try:
            brands = brands.drop(index=idx)
        except KeyError:
            pass

    count_cleaned = brands.sum()[0]
    percentage = 100 * brands / brands.sum()

    print('Značky jsou v {} % případů neznámé, nebo jinak nekonkrétně určené, tyto jsou dále ignorovány.' \
          .format(float(1 - count_cleaned / count_total) * 100))

    # tvorba grafu
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot()

    percentage[0:12].plot.bar(ax=ax,
                              legend=None,
                              color='#ffa600')
    f = mtick.FuncFormatter(lambda x, p: format(int(x)) + ' %')
    ax.yaxis.set_major_formatter(f)
    plt.gca().set_title('TOP 12 značek')
    ax.set_xlabel('')
    ax.xaxis.grid(False)
    plt.tight_layout()

    if fig_location:
        d = os.path.dirname(fig_location)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        fig.savefig(fig_location)

    if show_figure:
        fig.show()


def plot_brands_yearly(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    years = df['rok_nehody'].unique()
    for year in years:
        df_current = df[df['rok_nehody'] == year]

        brands = df_current.groupby(['znacka']) \
            .aggregate({'znacka': 'count'}) \
            .rename(columns={'znacka': 'pocet'}) \
            .sort_values(by='pocet', ascending=False)

        for i in [0, 98, 99]:
            # odstraneni zaznamu neuvedenych znacek
            idx = brands_number_to_name[i]
            try:
                brands = brands.drop(index=idx)
            except KeyError:
                pass

        count_cleaned = brands.sum()[0]
        percentage = 100 * brands / brands.sum()

        # tvorba grafu
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot()

        percentage[0:12].plot.bar(ax=ax,
                                  legend=None,
                                  color='#ffa600')

        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.gca().set_title('TOP 12 značek ROK XXXX')
        ax.set_xlabel('')
        ax.xaxis.grid(False)
        plt.tight_layout()

        if fig_location:
            d = os.path.dirname(fig_location)
            if d and not os.path.isdir(d):
                os.makedirs(d)
            fig.savefig(fig_location)

        if show_figure:
            fig.show()


if __name__ == '__main__':
    # global style
    sns.set_style('darkgrid')
    sns.set_context('notebook')

    """příprava dat"""

    # , druh vozidla, značka, rok výroby (19xx, nebo 20xx),                         # date, gps, gps
    df = pd.read_pickle('accidents.pkl.gz')[['p2a', 'p44', 'p45a', 'p47', 'p48a']]  # 'p1', 'd', 'e'
    count_total = len(df.index)

    df = df.rename(columns={
        'p2a': 'rok_nehody',
        'p44': 'druh',
        'p45a': 'znacka',
        'p47': 'rok_vyroby',
        'p48a': 'vlastnik'})

    df['znacka'] = df['znacka'].map(_brands_number_to_name)  # pozor, nepoužívá defaultdict

    df['rok_nehody'] = df['rok_nehody'].astype('datetime64[Y]').dt.year

    df['rok_vyroby'] = df['rok_vyroby'].astype('int')
    df = df[df['rok_vyroby'] != -1]

    df.loc[df['rok_vyroby'] > 20, 'rok_vyroby'] += 1900
    df.loc[df['rok_vyroby'] <= 20, 'rok_vyroby'] += 2000

    df['stari'] = df['rok_nehody'] - df['rok_vyroby']
    print('Nejstarší bourané auto bylo {} let staré.'.format(df['stari'].max()))

    """tvorba grafů"""
    # TOP N značek
    plot_brands_total(df, 'znacky_celkove.pdf', True)

    # Stáří aut
    plot_age_total(df, 'stari_celkove.pdf', True)

    plot_age_yearly(df, 'stari_v_letech.pdf', True)

    # saved
    # sns.pairplot(data=df, hue="rok_nehody")
    # plt.show()
