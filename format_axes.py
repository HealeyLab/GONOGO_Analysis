def format_ax(ax, **kwargs):
    kwargs_dict = dict(kwargs)
    if 'ylabel' in kwargs_dict:
        ax.set_ylabel(kwargs_dict['ylabel'])
    if 'xlabel' in kwargs_dict:
        ax.set_xlabel(kwargs_dict['xlabel'])
    ax.tick_params(axis='both', which='major')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')