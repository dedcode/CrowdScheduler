import os.path
import time
import gc
import pandas
import numpy as np
import matplotlib as mpl
mpl.use('pgf')
def figsize(scale):
    fig_width_pt = 517.935                         # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size
pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 8,               # LaTeX default is 10pt font.
    "text.fontsize": 8,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.5),     # default fig size of 0.5 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)
import matplotlib.pyplot as plt

# I make my own newfig and savefig functions
def newfig(width):
    plt.clf()
    fig = plt.figure(figsize=figsize(width))
    ax = fig.add_subplot(111)
    return fig, ax

def savefig(filename):
    plt.savefig('{}.pgf'.format(filename))
    plt.savefig('{}.pdf'.format(filename))

pandas.options.display.mpl_style = 'default'
plt.style.use('ggplot')


def main(**kwargs):
    options = {}
    options['filename'] = '2400agg'
    for key, value in kwargs.iteritems():
        options[key] = value
    try:
        r = pandas.read_pickle(options['filename']+'.pkl')
    except Exception as e:
        print(e)
        print('no file?', options['filename']+'.pkl')
        return
    r.rename(columns={'deadline_success_ratio': 'Dead.\,succ.\,ratio', 'mean_best_effort': 'Mean best effort',
                      'std_best_effort': 'Std best effort', 'std_deadline': 'Std deadline',
                      'scheduler': 'Scheduler'}, inplace=True)
    r.loc[r['Scheduler'] == 'fair', 'Scheduler'] = 'Fair'
    r.loc[r['Scheduler'] == 'saware', 'Scheduler'] = 'S.Awr'
    r.loc[r['Scheduler'] == 'aware', 'Scheduler'] = 'Aware'
    r.loc[r['Scheduler'] == 'deadline', 'Scheduler'] = 'SD'
    for s in r.strictness.unique():
        fig, ax = newfig(0.62)
        ax = r[(r['strictness'] == s)].boxplot(column=['Dead.\,succ.\,ratio', 'Std deadline', 'Mean best effort',
                                                  'Std best effort'], by=['Scheduler'], showfliers=False, ax=ax,
                                               notch=True, patch_artist=True)
        plt.suptitle("")
        # print(ax)
        # print(ax.__dict__)
        for i in range(0, 2):
            for j in range(0,2):
                ax[i, j].set_ylim([-0.05, 1.05])

        fig = ax[0][0].get_figure()
        fig.suptitle('')
        #print(fig.__dict__)
        #plt.savefig(options['filename']+'_'+str(s)+'_boxplot.eps')
        savefig(options['filename']+'_'+str(s)+'_boxplot')

if __name__ == "__main__":
    main()