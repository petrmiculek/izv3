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

"""



"""
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


def plot_age(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    df['datum'] = df['datum'].astype('datetime64[Y]').dt.year
    df['rok_vyroby'] = df['rok_vyroby'].astype('int')
    df = df[df['rok_vyroby'] != -1]

    count_valid_rok = len(df.index)  # 382k

    df.loc[df['rok_vyroby'] > 20, 'rok_vyroby'] += 1900
    df.loc[df['rok_vyroby'] <= 20, 'rok_vyroby'] += 2000

    age = df['datum'] - df['rok_vyroby']

    n = df['rok_vyroby'].unique()

    x, y = np.unique(age, return_counts=True)

    # data stačí v tisících
    # noinspection PyAugmentAssignment
    y = y / 1000

    max_age = 55

    fig = plt.figure(figsize=(6, 4))
    ax = sns.lineplot(x=x[:max_age], y=y[:max_age], color='#bc5090', marker='o')
    ax.set_title('Stáří aut [roků]')
    ax.set_ylabel('počet aut [v tisících]')
    ax.set_xlabel('roků')
    ax.set_xlim(right=42)
    fig.tight_layout()
    fig.show()

    if fig_location:
        d = os.path.dirname(fig_location)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        fig.savefig(fig_location)

    if show_figure:
        fig.show()


def plot_brands(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
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

    # kolacovy graf - nepouzito
    # percentage.plot.pie(y='pocet', legend=None)

    print('Značky jsou v {} % případů neznámé, nebo jinak nekonkrétně určené, tyto jsou dále ignorovány.' \
          .format(float(1 - count_cleaned / count_total) * 100))

    # tvorba grafu
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot()

    percentage[0:12].plot.bar(ax=ax,
                              legend=None,
                              color='#ffa600')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
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


if __name__ == '__main__':
    # global style
    sns.set_style('darkgrid')
    sns.set_context('notebook')

    # , druh vozidla, značka, rok výroby (XX),                                      # date, gps, gps,
    df = pd.read_pickle('accidents.pkl.gz')[['p2a', 'p44', 'p45a', 'p47', 'p48a']]  # 'p1', 'd', 'e'
    count_total = len(df.index)

    df = df.rename(columns={'p2a': 'datum', 'p44': 'druh', 'p45a': 'znacka', 'p47': 'rok_vyroby', 'p48a': 'vlastnik'})

    df['znacka'] = df['znacka'].map(_brands_number_to_name)
    # todo nakonec bez defaultdict

    # TOP N značek
    plot_brands(df, 'brands.pdf', True)

    # Stáří aut
    plot_age(df, 'age.pdf', True)
