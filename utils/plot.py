import seaborn as sns
import matplotlib.pyplot as plt


def plot_dist(noise_feats, label=None, ymax=1.0, color=None, title=None, save_path=None):
    sns.set_style('darkgrid')
    ax = sns.distplot(noise_feats, hist=False, kde=True, kde_kws={'label': label}, color=color)
    plt.xlim(-3, 11)
    plt.ylim(ymin=0.0, ymax=ymax)

    if title:
        plt.title(title)
        
    if save_path:
        plt.savefig(save_path)

    return ax
