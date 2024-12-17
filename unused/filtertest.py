from mne.viz import plot_filter
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt 
import mne

# freBand = {'none':[],'delta':[1,4],'theta':[4,8],\
#             'alpha':[8,12],'beta':[12,25],'lowgamma':[25,58]}
freqband = [4,8]
fs = 100

minfac = 3 # this many lowest frequency cycles in filter
# order=int(minfac*np.fix(fs/freqband[0]))
order=10
trans = 0.25 # fractional width of transition zones
# b = signal.firwin(order, freqband, width = trans*freqband[0], pass_zero=False,fs=fs)
# b = signal.firwin2(order, freq=[0,(1-trans)*freqband[0],freqband[0],freqband[1],freqband[1]*(1+trans),fs/2], gain=[0,0,1,1,0,0],window=('kaiser',14),fs=fs)
# b = signal.firls(order, [0,(1-trans)*freqband[0],freqband[0],freqband[1],freqband[1]*(1+trans),fs/2], desired=[0,0,1,1,0,0],fs=fs)
filter_params = mne.filter.create_filter(np.random.random((1,1000)), fs,
                                         l_freq=freqband[0], h_freq=freqband[1])
mne.viz.plot_filter(filter_params, fs)

# b,a = signal.butter(order, freqband, 'bandpass', fs=fs)
# w, h = signal.freqs(b, a)

fig1, axes = plt.subplots(2, 1)
w, h = signal.freqz(b)

axes[0].semilogy(0.5*fs*w/np.pi,np.abs(h))
axes[1].plot(0.5*fs*w/np.pi,np.unwrap(np.angle(h)))
t, h = signal.dimpulse((b,1))
plt.plot(t,h)