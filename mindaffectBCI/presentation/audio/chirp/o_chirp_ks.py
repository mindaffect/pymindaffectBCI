#  Copyright (c) 2021 MindAffect B.V. 
#  Author: Khash sharif <khash@mindaffect.nl>
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp,gausspulse,windows

# generate an O-chirp audio stimulus based on the method proposed by Torsten Dau(Nov 2004)
def O_chirp_ks(c=0.15, alpha=-0.5, t=0.0135, f0 = 100, f=10000, sps=44100, phase=0, normalize=True, plot=True,scipyChirp=True):
    """_summary_

    Args:
        c (float, optional): parameter of the chirp generator, see https://asa.scitation.org/doi/pdf/10.1121/1.1787523. Defaults to 0.15.
        alpha (float, optional): parameter of the chirp generator, see https://asa.scitation.org/doi/pdf/10.1121/1.1787523. Defaults to -0.5.
        t (float, optional): duration of the chirp in seconds. Defaults to 0.0135.
        f0 (int, optional): minimum frequency of the chirp.  Frequency at chirp start. Defaults to 100.
        f (int, optional): maximum frequency of the chirp. Frequency at chirp end. Defaults to 10000.
        Nsamples (int, optional): number of samples in 0-t seconds. Defaults to 596.
        phase (int, optional): _description_. Defaults to 0.
        normalize (bool, optional): _description_. Defaults to True.
        plot (bool, optional): if set to True, the chirp signal is plotted in time domain(Amplitude vs seconds). Defaults to True.
        scipyChirp (bool, optional): use scipy chirp generator, or our own one. Defaults to True.
    """
    Nsamples = int(t*sps)
    if scipyChirp:
        t0 = ((2*t)+ math.sqrt((4*t*t)- (4*((t*t)-(c*c/f)))))/2
        twave = np.linspace(0, t, Nsamples)
        S = chirp(twave, f0=f0, f1=f, t1=t, method='quadratic')

        # Amplitude modulation 
        # gaussian window to suppress the edges, but biased to slowly increase and rapidly decrease
        mu, sigma = int(Nsamples*.75), .5
        w = np.concatenate( (np.exp( -.5 * (np.arange(0,mu) - mu)**2 / (sigma*mu)**2),
                      np.exp( -.5 * (np.arange(mu,Nsamples) - mu)**2 / (sigma*(Nsamples-mu))**2)) )
        # zero the edges
        w = w - np.min(w)
        # increase amplitude with frequency to compensate for reduced frequency response
        # w = \sqrt(2c^2)/\sqrt((t-t0)^3) ~= 1/(t^3) => suppress the initial part of the wave
        #w2 = [ math.sqrt(2*c*c/((t0-t)**2)) for t in twave]
        #w= gausspulse(twave, fc=500, bw=0.5, bwr=- 6, tpr=- 60, retquad=False, retenv=False)
        # combine the windows
        #w = [ w1*w2 for w1,w2 in zip(w,w2)]

        # apply the am-modulator to the chirp
        S = S * w
    else:
        t0 = ((2*t)+ math.sqrt((4*t*t)- (4*((t*t)-(c*c/f)))))/2
        fmax=(pow(c/(t0-t),2))
        twave = np.linspace(0, t, Nsamples)
        S=np.zeros(len(twave))
        for i in range (len(twave)):
            S[i]=math.sin(2*math.pi*c*c*((1/(t0-twave[i]))-1/t0) - phase) * (math.sqrt(2*c*c/(math.pow(t0-twave[i],3))))
    if normalize:
        S=S/(max(S))
    if plot:
        plt.plot(twave, S)
        plt.show()
    return S


def make_chirp_wav(t,f,f0,sps=44100):
    signal = O_chirp_ks(t=t, f0=f0, f=f, sps=sps)
    # Write the .wav file
    import os
    fname = os.path.join(os.path.dirname(__file__),'{}-{}-gauss.wav'.format(f0,f))
    print('Saving to: {}'.format(fname))

    from scipy.io.wavfile import write
    # N.B. convert to int to ensure is stored as int-wav format
    amp = np.iinfo(np.int16).max*.99
    signal = (signal * amp).astype(np.int16)
    write(fname, sps, signal)
    return signal


if __name__ == "__main__":
    # make chirps
    t=0.033 # chrip duration = 1/30s
    chirps = [(150,350),(400,600),(800,1200),(1600,2400),(3000,5000),(5500,6500),(7250,9000)]
    for f0,f in chirps:
        make_chirp_wav(t,f,f0)