import mapclassify
import numpy as np
from numba import jit
import warnings
from math import pi
import matplotlib.pyplot as plt
@jit
def _kclasses(arr,k=5):
    kclasses=mapclassify.FisherJenks(arr,k=k)
    kclasses=kclasses.make()(arr)
    return kclasses

def linewidths_by_attribute_fisherjenks(gdf,attr,k=5,vmin=.5,vmax=6):
    arr = np.array(gdf[attr])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kclasses = _kclasses(arr,k=k)
    s = sorted(set(kclasses))
    sizes = [(vmin+n*((vmax-vmin)/(k-1))) for n in s]
    d = {n:m for n,m in zip(s,sizes)}
    return [d[n] for n in kclasses]

def plot_radar(values,labels,rotation_degrees=0,
               xtick_color='k',ytick_color='k',
               line_color='grey',fill_color='grey',
               alpha=.15,linewidth=1,linestyle='solid',
               label=None):

    N = len(labels)
 
    # repeat the first value to close the circular graph:
    values += values[:1]

    # Angle of each axis
    rot = rotation_degrees*pi/180
    angles = [rot + n/float(N)*2*pi for n in range(N)]
    angles += angles[:1]
    for i in range(len(angles)):
        if angles[i]>2*pi or angles[i]<0:
            angles[i]=angles[i]-2*pi*(angles[i]//(2*pi))
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], labels, color=xtick_color)

    # Draw ylabels
    ax.set_rlabel_position(90)
    plt.yticks([.2,.4,.6,.8,1], [f'{n*100:.00f}%' for n in [.2,.4,.6,.8,1]], color=ytick_color,rotation='vertical')
    plt.ylim(0,1)

    # Plot data
    ax.plot(angles, values, linewidth=linewidth, linestyle=linestyle,color=line_color,label=label)

    # Fill area
    ax.fill(angles, values, color=fill_color, alpha=alpha)

    # Show the graph
    return ax