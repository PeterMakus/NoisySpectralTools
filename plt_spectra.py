'''
Simple script to compute and plot time-dependent spectral power densities.

Author: Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 15th February 2021 02:09:48 pm
Last Modified: Wednesday, 3rd March 2021 12:38:47 pm
'''
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
import numpy as np
from obspy import read, UTCDateTime
import obspy
from obspy.clients.fdsn import Client
from scipy.signal import welch
from scipy.interpolate import pchip_interpolate


def main():
    # read waveform data
    os.chdir('/home/chris/PROJECTS/KISS/DATA/mseed')
    
    for folder, _, _ in os.walk('.'):
        try:
            # If this becomes to RAM hungry, I might want to load each
            # file and compute seperately
            st = read(os.path.join(folder, '*.shz'))
        except FileNotFoundError:
            continue
        except Exception:
            # They have a different Exception for file patterns
            continue
        name = '%s.%s_spectrum_normf' %(st[0].stats.network, st[0].stats.station)
        outf = os.path.join('/home', 'makus', 'samovar', 'figures',
                            'spectrograms', name)
        
        if  Path(outf+'.npz').is_file():
            with np.load(outf+'.npz') as A:
                l = []
                for item in A.files:
                    l.append(A[item])
                f, t, S = l
                # plot
                plot_spct_series(S, f, t, title=name, outfile=outf, norm='f')
                plt.savefig(outf+'.png', format='png', dpi=300)
                # just in case
                plt.close()
        else:
            # preprocess the data
            st, _ = preprocess(st)
            # compute a spectral series with 4-hourly spaced data points
            f, t, S = spct_series_welch(st, 4*3600)

            # Save to file
            np.savez(outf, f, t, S)

            # plot
            plot_spct_series(S, f, t, title=name, outfile=outf)
            plt.savefig(outf+'.png', format='png', dpi=300)
            # just in case
            plt.close()
        
            
def plot_spct_series(
    S:np.ndarray, f:np.ndarray, t:np.ndarray, norm:str=None, title:str=None,
    outfile=None, fmt='pdf', dpi=300):
    """
    Plots a spectral series.

    :param S: A spectral series with dim=2.
    :type S: np.ndarray
    :param f: Vector containing frequencies (Hz)
    :type f: np.ndarray
    :param t: Vector containing times (in s)
    :type t: np.ndarray
    :param norm: Normalise the spectrum either on the time axis with
        norm='t' or on the frequency axis with norm='f', defaults to None.
    :type norm: str, optional
    :param title: Plot title, defaults to None
    :type title: str, optional
    :param outfile: location to save the plot to, defaults to None
    :type outfile: str, optional
    :param fmt: Format to save figure, defaults to 'pdf'
    :type fmt: str, optional
    """
    # Create UTC time series
    utc = []
    for pit in t:
        utc.append(UTCDateTime(pit).datetime)
    del t
    
    set_mpl_params()
    
    plt.yscale('log')
    plt.ylim(10**-2, f.max())
    
    # Normalise
    if not norm:
        pass
    elif norm == 'f':
        S = np.divide(S, np.linalg.norm(S, axis=0))
    elif norm == 't':
        S = np.divide(S, np.linalg.norm(S, axis=1))
    else:
        raise ValueError('Normalisation '+norm+' unkown.')
    
    pcm = plt.pcolormesh(
        utc, f, S, shading='gouraud', norm=colors.LogNorm(vmin=S.min(), vmax=S.max()))
    plt.colorbar(pcm)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('(dd/mm)')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    if title:
        plt.title(title)
    if outfile:
        if fmt=='pdf' or fmt=='svg':
            plt.savefig(outfile+'.'+fmt,format=fmt)
        else:
            plt.savefig(outfile+'.'+fmt,format=fmt, dpi=dpi)
    else:
        plt.show()    

def spct_series_welch(st:obspy.Stream, window_length:int or float):
    """
    Computes a spectral time series. Each point in time is computed using the
    welch method. Windows overlap by half the windolength. The input stream can
    contain one or several traces from the same station. Frequency axis is
    logarithmic.

    :param st: Input Stream with data from one station.
    :type st: ~obspy.core.Stream
    :param window_length: window length in seconds for each datapoint in time
    :type window_length: int or float
    :return: Arrays containing a frequency and time series and the spectral series.
    :rtype: np.ndarray
    """
    l = []
    st.sort(keys=['starttime'])
    for tr in st:
        for wintr in tr.slide(window_length=window_length, step=window_length):
            # windows will overlap with half the window length
            # Hard-corded nperseg so that the longest period waves that
            # can be resolved are around 300s
            f, S = welch(wintr.data, fs=tr.stats.sampling_rate, nperseg=2**15)
            
            # interpolate onto a logarithmic frequency space
            # 256 points of resolution in f direction hardcoded for now
            f2 = np.logspace(-3, np.log10(f.max()), 256)
            S2 = pchip_interpolate(f, S, f2)
            l.append(S2)
    S = np.array(l)
    
    # compute time series
    t = np.linspace(
        st[0].stats.starttime.timestamp, st[-1].stats.endtime.timestamp,
        S.shape[0])
    return f2, t, S.T     
        
def preprocess(st:obspy.Stream):
    """
    Some very basic preprocessing on the string in order to plot the spectral
    series. Does the following steps:
    *1. Remove station response*
    *2. Detrend*
    *3. Decimate if sampling rate>50*
    *4. highpass filter with corner period of 300s.*
    *5. Saves the processed traces (and disregards already processed ones
    in first place.)*

    :param st: The input Stream, should only contain Traces from one station.
    :type st: ~obspy.core.Stream
    :return: The output stream and station inventory object
    :rtype: ~obspy.core.Stream and ~obspy.core.Inventory
    """
           
    # Download station responses
    gfz = Client('GFZ')
    inv = gfz.get_stations(
        network=st[0].stats.network,station=st[0].stats.station,
        channel=st[0].stats.channel, level='response')
    l = []
    for tr in st:
        loc = os.path.join(
            '/home','makus', 'samovar', 'data', 'preprocessed',
            '%s.%s_%s.mseed' %(tr.stats.network, tr.stats.station, tr.stats.starttime))
        # Was file already preprocessed?
        if Path(loc).is_file():
            tr = read(loc)[0]
            l.append(tr)
            del st[0]
            continue
        
        # Downsample to make computations faster
        if tr.stats.sampling_rate < 50:
            tr.decimate(2)
        # Remove station responses
        tr.attach_response(inv)
        tr.remove_response()
        
        # Detrend
        tr.detrend()

        # highpass filter
        tr.filter('highpass', freq=1/300, zerophase=True)
        
        # Save preprocessed stream
        tr.write(loc, format='MSEED')
        l.append(tr)
        del st[0]
    return obspy.Stream(l), inv

def set_mpl_params():
    params = {
        #'font.family': 'Avenir Next',
        'pdf.fonttype': 42,
        'font.weight': 'bold',
        'figure.dpi': 150,
        'axes.labelweight': 'bold',
        'axes.linewidth': 1.5,
        'axes.labelsize': 14,
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 13,
        'xtick.direction': 'in',
        'xtick.top': True,  # draw label on the top
        'xtick.bottom': True,  # draw label on the bottom
        'xtick.minor.visible': True,
        'xtick.major.top': True,  # draw x axis top major ticks
        'xtick.major.bottom': True,  # draw x axis bottom major ticks
        'xtick.minor.top': True,  # draw x axis top minor ticks
        'xtick.minor.bottom': True,  # draw x axis bottom minor ticks
        'ytick.labelsize': 13,
        'ytick.direction': 'in',
        'ytick.left': True,  # draw label on the top
        'ytick.right': True,  # draw label on the bottom
        'ytick.minor.visible': True,
        'ytick.major.left': True,  # draw x axis top major ticks
        'ytick.major.right': True,  # draw x axis bottom major ticks
        'ytick.minor.left': True,  # draw x axis top minor ticks
        'ytick.minor.right': True,  # draw x axis bottom minor ticks
        'legend.fancybox': False,
        'legend.frameon': False,
        'legend.loc': 'upper left',
        'legend.numpoints': 2,
        'legend.fontsize': 'large',
        'legend.framealpha': 1,
        'legend.scatterpoints': 3,
        'legend.edgecolor': 'inherit'
    }
    matplotlib.rcParams.update(params)
    matplotlib.font_manager._rebuild()


main()

