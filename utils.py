import os
import numpy as np
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors


def check_dir(fn):
    if not os.path.exists(fn):
        os.makedirs(fn)


def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
        lr +=[ param_group['lr'] ]
    assert(len(lr)==1)

    return lr[0]


def adjust_colorbar(fig, axes, pad=0.02, width=0.02, shrink=0.8, xscale=0.95):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if isinstance(axes, np.ndarray):
        cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])

        fig.canvas.draw()
        pos = cax.get_position()
    elif isinstance(axes, mpl.axes.Axes):
        divider = make_axes_locatable(axes)
        cax = divider.append_axes('right', size='5%', pad='3%')
        kw = {'orientation': 'vertical', 'ticklocation': 'right'}

        fig.canvas.draw()
        pos = axes.get_position()
    else:
        raise TypeError(f'No support axes type {type(ax)}!')

    ydf = (1-shrink)*(pos.ymax - pos.ymin)/2
    cax.remove()

    return fig.add_axes([pos.xmax*xscale+pad, pos.ymin+ydf, width, (pos.ymax-pos.ymin)-2*ydf]), kw


def array2img(array, cmap, vmin=0, vmax=70, rgb=False):
    """
    :params array(np.array):
    :params cmap(str): colormap name
    :params vmin(int): vmin for normalize
    :params vmax(int): vmax for normalize
    :params rgb(bool): if convert image to rgb format
    Ref:
      - https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
    """
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    if isinstance(cmap, str):
        cms = cm.get_cmap(cmap)
    elif isinstance(cmap, list) or isinstance(cmap, np.ndarray):
        cms = colors.LinearSegmentedColormap.from_list('cmap', cmap)
    else:
        raise ValueError(f'Unknown type {type(cmap)}.')

    if rgb:
        return Image.fromarray(np.uint8(cms(norm(array))*255)).convert('RGB')
    else:
        return Image.fromarray(np.uint8(cms(norm(array))*255))


def save_pred(pred, save_path, configs=None):
    """
    :param pred(np.array): prediction
    :param save_path(str): the path to save prediction
    """
    if pred.ndim != 3:
        raise ValueError(f'The dimension of prediction should be 3, not {pred.ndim}!')

    for ti in range(pred.shape[0]):
        fig, axes = plt.subplots(figsize=(9, 6))
        fig, axes = single_plot(pred[ti], ts=ti, fig=fig, axes=axes, title=True)
        fig.savefig(f'{save_path}{os.sep}{ti:02d}.png', dpi=100, bbox_inches='tight')


def single_plot(pred, target=None, ts=0, fig=None, axes=None, title=True):
    import seaborn as sns

    try:
        import pyart
        cmap = 'pyart_NWSRef'
    except:
        colev = ["#99DBEA", "#52A5D1", "#3753AD", "#80C505", "#52C10D",
                "#35972A", "#FAE33B", "#EAB81E", "#F78C2C", "#E2331F",
                "#992B27", "#471713", "#BC5CC2", "#975CC0"]

        cmap = colors.ListedColormap(colev)

        levels = np.arange(0, 71, 5)
        norm = colors.BoundaryNorm(levels, cmap.N)

    sns.set_context('talk', font_scale=0.8)

    if target is not None:
        if fig is None and axes is None:
            fig, axes = plt.subplots(figsize=(12, 9), nrows=1, ncols=2)
            plt.subplots_adjust(wspace=0.05, hspace=0.05)

        im1 = axes[0].imshow(target, vmin=0, vmax=70, cmap=cmap)
        im2 = axes[1].imshow(pred, vmin=0, vmax=70, cmap=cmap)

        for i in range(axes.size):
            axes[i].set_axis_off()

        if title:
            axes[0].set_title(f'Target {ts}')
            axes[1].set_title(f'Prediction {ts}')

        cax, kw = adjust_colorbar(fig, axes, shrink=0.53, xscale=0.91)

        cb = fig.colorbar(im2, cax=cax, **kw)
        cb.set_ticks(np.arange(0, 71, 10))
        #cb.set_ticklabels(np.arange(0, 71, 10))
        cb.ax.tick_params(direction='in', left=True, right=True, length=3,
                          axis='both', which='major', labelsize=14)
    else:
        fig, axes = plt.subplots(figsize=(9, 6))
        im1 = axes.imshow(pred, vmin=0, vmax=70, cmap=cmap)

        axes.set_axis_off()

        fig.colorbar(im1)

    return fig, axes


def plot_compare(targets, prec, epoch=None, save_path=None):
    if epoch is None:
        epoch = 'default'
    if save_path is None:
        save_path = '.'

    for ti in range(targets.shape[0]):
        fig, axes = plt.subplots(figsize=(12, 9), nrows=1, ncols=2)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        fig, axes = single_plot(prec[ti], target=targets[ti], ts=ti, fig=fig, axes=axes, title=True)

        fig.savefig(f'{save_path}{os.sep}radar_{epoch}_time{ti:02d}.png', dpi=100, bbox_inches='tight')

    return fig, axes

