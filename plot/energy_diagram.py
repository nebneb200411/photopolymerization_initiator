import matplotlib.pyplot as plt
import numpy as np

def singlet_and_triplet(singlets, triplets, name=None, save=False):
    """一重項と三重厚のエネルギーダイアグラムを作成

    Args
        singlets: 一重項のエネルギーリスト
        triplets: 三重厚のエネルギーリスト
    
    Returns
        エネルギーダイアグラム
    """
    singlets.sort()
    triplets.sort()

    fig, ax = plt.subplots(figsize=(10, 10))

    # plot singlets
    ax.scatter(np.full(len(singlets), 1), singlets, s=1444, marker="_", linewidth=3, zorder=3)
    # plot triplets
    ax.scatter(np.full(len(triplets), 3), triplets, s=1444, marker="_", linewidth=3, zorder=3)

    singlets_labels = ['$S_{}$'.format(x+1) for x in range(len(singlets))]
    triplets_labels = ['$T_{}$'.format(x+1) for x in range(len(triplets))]

    for i, label, energy in zip(range(len(singlets)), singlets_labels, singlets):
        if i % 2 == 0:
            ax.annotate(label, xy=(1, energy), xytext=(-40, 0), size=12, textcoords="offset points")
            ax.annotate(str(energy), xy=(1, energy), xytext=(-10, 0), size=8, textcoords="offset points")
        if i % 2 != 0:
            ax.annotate(label, xy=(1, energy), xytext=(40, 0), size=12, textcoords="offset points")
            ax.annotate(str(energy), xy=(1, energy), xytext=(-10, 0), size=8, textcoords="offset points")

    for i, label, energy in zip(range(len(triplets)), triplets_labels, triplets):
        if i % 2 == 0:
            ax.annotate(label, xy=(3, energy), xytext=(-40, 0), size=12, textcoords="offset points")
            ax.annotate(energy, xy=(3, energy), xytext=(-10, 0), size=8, textcoords="offset points")
        if i % 2 != 0:
            ax.annotate(label, xy=(3, energy), xytext=(40, 0), size=12, textcoords="offset points")
            ax.annotate(energy, xy=(3, energy), xytext=(-10, 0), size=8, textcoords="offset points")
    
    ax.annotate('$S_{0}$', xy=(1, 0), xytext=(-4, 0), size=12, textcoords="offset points")
    ax.set_xlim(0, 4)
    ax.set_xticks([1, 3])
    ax.set_xlabel("Multiplicity")
    ax.set_ylabel("Energy [eV]")

    plt.show()

    if save and name:
        fig.savefig('./result/{}_singlets_triplets.png'.format(name))
