import matplotlib.pyplot as plt


def plot_eigs(eigs, **kwargs):
    """
    Plot the provided eigenvalues (of the dynamics operator A).
    
    Args:
        eigs (:obj:`ndarray` of complex): 
        **kwargs: kwargs of matplotlib.pyplot.subplots

    Returns:
        (tuple): Tuple containing:
            fig: figure object
            ax: axes object
    """
    xlim = kwargs.pop('xlim', [-1.1, 1.1])
    ylim = kwargs.pop('xlim', [-1.1, 1.1])

    fig, ax = plt.subplots(1, **kwargs)
    ax.set_aspect('equal'), ax.set_xlim(xlim), ax.set_ylim(ylim)
    ax.scatter(eigs.real, eigs.imag)
    ax.add_artist(plt.Circle((0, 0), 1, color='k', linestyle='--', fill=False))
    return fig, ax

# TODO: def hinton(args):