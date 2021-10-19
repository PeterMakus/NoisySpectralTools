'''
Simple script to compute and plot time-dependent spectral power densities.

Author: Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 15th February 2021 02:09:48 pm
Last Modified: Tuesday, 19th October 2021 01:34:17 pm
'''
import os
from typing import List, Tuple
import warnings
import logging
from datetime import datetime
import locale

from mpi4py import MPI
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
import numpy as np
from obspy import read, UTCDateTime, read_inventory
import obspy
from obspy.clients.fdsn import Client
from obspy.core.stream import Stream
from scipy.signal import welch
from scipy.interpolate import pchip_interpolate
from seismic.utils.miic_utils import resample_or_decimate
from seismic.trace_data.waveform import Store_Client


def main():
    # init mpi
    comm = MPI.COMM_WORLD
    psize = comm.Get_size()
    rank = comm.Get_rank()
    # Let's get rid of all the annoying mseed warnings
    warnings.filterwarnings("ignore")
    # read waveform data
    os.chdir('/home/makus/samovar/data/mseed')
    client = 'GFZ'
    norm = 'f'
    norm_meth = 'median'
    tlim = (datetime(2016, 1, 15), datetime(2016, 4, 1))
    flim = (0.5, 2)
    sc = Store_Client(client, '/home/makus/samovar/data', read_only=True)
    if rank == 0:
        statlist = sc.get_available_stations()
    else:
        statlist = None
    statlist = comm.bcast(statlist, root=0)

    # do the mpi stuff
    pmap = np.arange(len(statlist))*psize/len(statlist)
    pmap = pmap.astype(np.int32)
    ind = pmap == rank
    ind = np.arange(len(statlist))[ind]
    for net, stat in np.array(statlist)[ind]:
        name = '%s.%s_spectrum' % (
            net, stat)
        outf = os.path.join(
            '/home', 'makus', 'samovar', 'figures', 'spectrograms_N', name)
        outfig = '%s%s_%s_%s' % (
            outf, norm_meth, str(flim) or '', str(tlim) or '') 
        try:
            with np.load(outf + '.npz') as A:
                l = []
                for item in A.files:
                    l.append(A[item])
                f, t, S = l
                # plot
            plot_spct_series(
                S, f, t, title=name, outfile=outfig, norm=norm,
                norm_method=norm_meth, flim=flim, tlim=tlim)
            plt.tight_layout()
            plt.savefig(outfig+'.png', format='png', dpi=300)
            plt.close()
            continue
        except FileNotFoundError:
            pass
        except Exception as e:
            logging.exception(e)
            # just in case
            pass
        try:
            # Got to load the raw data then
            start, end = sc._get_times(net, stat)
            # loading everything at once takes way too much RAM
            # let's do daily chunks that are then saved to disk
            # and loaded by the next function again
            endtmp = end
            starttmp = start
            starts = [start]
            while end-starttmp > 24*3600:
                endtmp = starttmp + 24*3600
                preprocess(
                    sc._load_local(
                        net, stat, '*', '??N', starttmp, endtmp, False, False),
                    client, starttmp)
                starttmp = endtmp
                starts.append(starttmp)
            preprocess(
                sc._load_local(
                    net, stat, '*', '??N', starttmp, end, False, False),
                client, starttmp)

            # compute a spectral series with 4-hourly spaced data points
            f, t, S = spct_series_welch(starts, 4*3600, net, stat)

            # Save to file
            np.savez(outf, f, t, S)

            # plot
            plot_spct_series(
                S, f, t, title=name, outfile=outfig, flim=flim, norm=norm,
                norm_method=norm_meth, tlim=tlim)
            plt.savefig(outfig+'.png', format='png', dpi=300)
            # just in case
            plt.close()
        except Exception as e:
            logging.exception(e)


def plot_spct_series(
    S: np.ndarray, f: np.ndarray, t: np.ndarray, norm: str = None,
    norm_method: str = None, title: str = None, outfile=None, fmt='pdf',
    dpi=300, flim: Tuple[int, int] = None,
        tlim: Tuple[datetime, datetime] = None):
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
    :param norm_method: Normation method to use.
        Either 'linalg' (i.e., length of vector),
        'mean', or 'median'.
    :param title: Plot title, defaults to None
    :type title: str, optional
    :param outfile: location to save the plot to, defaults to None
    :type outfile: str, optional
    :param fmt: Format to save figure, defaults to 'pdf'
    :type fmt: str, optional
    :param flim: Limit Frequency axis and Normalisation to the values
        in the given window
    :type flim: Tuple[int, int]
    :param tlim: Limit time axis to the values in the given window
    :type tlim: Tuple[datetime, datetime]
    """
    # Show dates in English format
    locale.setlocale(locale.LC_ALL, "en_GB.utf8")
    # Create UTC time series
    utc = []
    for pit in t:
        utc.append(UTCDateTime(pit).datetime)
    del t

    set_mpl_params()

    plt.yscale('log')

    if flim is not None:
        plt.ylim(flim)
        ii = np.argmin(abs(f-flim[0]))
        jj = np.argmin(abs(f-flim[1])) + 1
        f = f[ii:jj]
        S = S[ii:jj, :]
    else:
        plt.ylim(10**-2, f.max())

    if tlim is not None:
        plt.xlim(tlim)
        utc = np.array(utc)
        ii = np.argmin(abs(utc-tlim[0]))
        jj = np.argmin(abs(utc-tlim[1]))
        utc = utc[ii:jj]
        S = S[:, ii:jj]

    # Normalise
    if not norm:
        pass
    elif norm == 'f':
        if norm_method == 'linalg':
            S = np.divide(S, np.linalg.norm(S, axis=1)[:, np.newaxis])
        elif norm_method == 'mean':
            S = np.divide(S, np.mean(S, axis=1)[:, np.newaxis])
        elif norm_method == 'median':
            S = np.divide(S, np.median(S, axis=1)[:, np.newaxis])
        else:
            raise ValueError('Normalisation method %s unkown.' % norm_method)
    elif norm == 't':
        if norm_method == 'linalg':
            S = np.divide(S, np.linalg.norm(S, axis=0))
        elif norm_method == 'mean':
            S = np.divide(S, np.mean(S, axis=0))
        elif norm_method == 'median':
            S = np.divide(S, np.median(S, axis=0))
        else:
            raise ValueError('Normalisation method %s unkown.' % norm_method)
    else:
        raise ValueError('Normalisation %s unkown.' % norm)

    pcm = plt.pcolormesh(
        utc, f, S, shading='gouraud',
        norm=colors.LogNorm(vmin=S.min(), vmax=S.max()))
    plt.colorbar(pcm)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('(dd/mm)')
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %h'))
    plt.xticks(rotation='vertical')
    if title:
        plt.title(title)
    if outfile:
        plt.tight_layout()
        if fmt == 'pdf' or fmt == 'svg':
            plt.savefig(outfile + '.' + fmt, format=fmt)
        else:
            plt.savefig(outfile+'.'+fmt, format=fmt, dpi=dpi)
    else:
        plt.show()


def spct_series_welch(
        starts: List[UTCDateTime], window_length: float, net: str, stat: str):
    """
    Computes a spectral time series. Each point in time is computed using the
    welch method. Windows overlap by half the windolength. The input stream can
    contain one or several traces from the same station. Frequency axis is
    logarithmic.

    :param st: Input Stream with data from one station.
    :type st: ~obspy.core.Stream
    :param window_length: window length in seconds for each datapoint in time
    :type window_length: int or float
    :return: Arrays containing a frequency and time series and the spectral
        series.
    :rtype: np.ndarray
    """
    l = []
    for start in starts:
        # windows will overlap with half the window length
        # Hard-corded nperseg so that the longest period waves that
        # can be resolved are around 300s
        loc = os.path.join(
            '/home','makus', 'samovar', 'data', 'preprocessed',
            '%s.%s_%s.mseed' % (
                net, stat, start))
        tr = read(loc)[0]
        for wintr in tr.slide(window_length=window_length, step=window_length):
            f, S = welch(wintr.data, fs=tr.stats.sampling_rate)
            # interpolate onto a logarithmic frequency space
            # 256 points of resolution in f direction hardcoded for now
            f2 = np.logspace(-3, np.log10(f.max()), 256)
            S2 = pchip_interpolate(f, S, f2)
            l.append(S2)
    S = np.array(l)
    
    # compute time series
    # t = np.linspace(
    #     st[0].stats.starttime.timestamp, st[-1].stats.endtime.timestamp,
    #     S.shape[0])
    t = np.linspace(
        starts[0].timestamp, starts[-1].timestamp, S.shape[0]
    )
    return f2, t, S.T


def preprocess(st: obspy.Stream, client, start: UTCDateTime):
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
    st_out = Stream()
    for tr in st:
        tr1 = tr.copy()
        loc = os.path.join(
            '/home','makus', 'samovar', 'data', 'preprocessed',
            '%s.%s_%s.mseed' % (
                tr.stats.network, tr.stats.station, start))
        # Was file already preprocessed?
        if os.path.isfile(loc):
            return
        # Station response already available?
        try:
            # inv = read_inventory(
            #     '/home/makus/samovar/data/inventory/%s.%s.xml' % (
            #         tr.stats.network, tr.stats.station))
            inv = read_inventory(
                '/home/makus/samovar/data/inventory/inventory.xml'
            )
        except FileNotFoundError:
            # Download station responses
            try:
                if isinstance(client, str):
                    client = Client(client)
                    client.set_eida_token('/home/makus/.eidatoken')
                inv = client.get_stations(
                    network=st[0].stats.network,station=st[0].stats.station,
                    channel='%s?' % tr.stat.channel[:-1], level='response')
                inv.write(
                    '/home/makus/samovar/data/inventory/%s.%s.xml' % (
                    tr.stats.network, tr.stats.station), format="STATIONXML")
            except Exception:
                # Still better to just proceed
                logging.exception('could not download inv')
                pass
        # Downsample to make computations faster
        # if tr.stats.sampling_rate > 50:
        #     tr.decimate(2)
        tr = resample_or_decimate(tr, 25)
        # Remove station responses
        try:
            tr.attach_response(inv)
            tr.remove_response()
        except ValueError:
            try:
                # Download station responses
                if isinstance(client, str):
                    client = Client(client)
                    client.set_eida_token('/home/makus/.eidatoken')
                inv = client.get_stations(
                    network=st[0].stats.network,station=st[0].stats.station,
                    channel='%s?' % tr.stat.channel[:-1], level='response')
                inv.write(
                    '/home/makus/samovar/data/inventory/%s.%s.xml' % (
                    tr.stats.network, tr.stats.station), format="STATIONXML")
                tr.attach_response(inv)
                tr.remove_response()
            except Exception:
                logging.exception('Could not remove instrument response')
                pass
        
        # Detrend
        tr.detrend(type='linear')

        # highpass filter
        # tr.filter('highpass', freq=1/300, zerophase=True)
        tr.filter('bandpass', freqmin=0.01, freqmax=12)

        # Save preprocessed stream
        
        st_out.append(tr)
    if st_out.count():
        st_out.write(loc, format='MSEED')


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
    # matplotlib.font_manager._rebuild()


main()

