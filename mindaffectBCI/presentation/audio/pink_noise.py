import numpy as np
import math
import cmath
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.io import wavfile
#import pyloudnorm as pyln
# adapted from https://scicomp.stackexchange.com/questions/18987/algorithm-for-high-quality-1-f-noise
def pink_noise(f_ref, f_min, f_max, length, f_sample):
    aliasfil_len = 10000
    fil_Time = aliasfil_len / f_sample
    L = length / f_sample + 2 * fil_Time
    f_low = 1 / L
    f_high = f_sample
    T = f_low * 2 * np.pi
    k_max = int(f_high / f_low / 2) + 1
    print(k_max)

    # Create frequencies
    f = np.array([(k * T)/(2 * np.pi) for k in range(0, k_max)])

    # Create 1/f noise amplitude in band
    C = np.array([(1 / f[k] if (f[k] >= f_min and f[k] <= f_max) else 0)
                  for k in range(0, k_max)], dtype='complex')
    C[0] = 0
    # Create random phase in band
    Phase = np.array([(np.random.uniform(0, np.pi)
                       if (f[k] >= f_min and f[k] <= f_max)
                       else 0)
                      for k in range(0, k_max)])

    Clist_neg = list()
    Clist_pos = list()
    for k in range(-k_max + 1, -1):
        Clist_neg.append(C[-k] * cmath.exp(-1j * Phase[-k]))
    for k in range(0, k_max):
        Clist_pos.append(C[k]  * cmath.exp( 1j * Phase[k] ))

    CC = np.array(Clist_pos + Clist_neg, dtype='complex')

    # Scale to max amplitude
    maxampl = max(abs(CC))
    CC /= maxampl

    tsig = np.fft.ifft(CC)
    sig = np.real(np.sign(tsig)) * np.abs(tsig)

    # Filter aliassing
    sig = sig[aliasfil_len:-aliasfil_len]

    # clip to maximum signal and
    # correct for amplitude at reference frequency
    if f_ref > ((f_max + f_min) / 2):
        print("WARNING: f_ref ({} Hz) should be smaller or equal to the mid "
              "between {} Hz and {} Hz "
              "to prevent clipping.\n"
              "f_ref changed to {} Hz"
              .format(f_ref,
                      f_min,
                      f_max,
                      ((f_max + f_min) / 2)))
        f_ref = ((f_max + f_min) / 2)
    maxampl = max(np.abs(sig))
    sig = sig / maxampl * f_ref / ((f_max + f_min) / 2)

    halfway = int(len(sig) / 2)
    # sign invert second part for a good connection,
    # it is the mirror of the first half
    sig2nd = -1 * sig[halfway:]
    sigc = np.concatenate((sig[0:halfway], sig2nd))
    # average middle point, but second point sign inverted
    sigc[halfway] = (sig[halfway-1] - sig[halfway+1])/2

    return(sigc)

def add_mixed_noise(signal,f_ref, f_min, f_max, length, f_sample, SNR=1, amplification_factor=1, signal_noise_amplitude_match=True, frame_size=1000):
    
    noise = pink_noise(f_ref, f_min, f_max, length, f_sample)
    len=min(noise.shape[0],signal.shape[0])
    if(signal_noise_amplitude_match):
        noise= noise/(max(abs(noise)))
        amp_envelope = Extract_multiple_AE(signal,frame_size)
        if amp_envelope.ndim >1:
            amp_envelope = amp_envelope.mean(axis=1)
        print(amp_envelope.shape)
        for i in range (0,len, frame_size):
            noise[i:i+frame_size]=noise[i:i+frame_size]*amp_envelope[int(i/frame_size)]
        #plt.plot(amp_envelope)
        #plt.show()
    #plt.plot((1-(1/(SNR+1)))*signal)
    print(signal[0:len].shape)
    if signal.ndim>1:
        for i in range(signal.shape[1]):
            signal[:,i] = amplification_factor*((1-(1/(SNR+1)))*signal[0:len,i] + (1/(SNR+1))*noise[0:len])
    else:
        signal = amplification_factor*((1-(1/(SNR+1)))*signal[0:len] + (1/(SNR+1))*noise[0:len])
    
    #plt.plot((1/(SNR+1))*noise[0:len])
    #plt.show()
    

    return(signal)


# method for extracting amplitude envelope of a signal
def Extract_AE(signal, frame_size):
    length = signal.shape[0]
    amp_envelope = []
    for i in range(0,length,frame_size):
        current_amp = max(signal[i:i+frame_size])
        amp_envelope.append(current_amp)
    return(np.array(amp_envelope))
    
def Extract_multiple_AE(signals, frame_size):
    ndim = signals.ndim
    if ndim == 1:
        amp_envelopes = Extract_AE(signals,frame_size)
    else:
        amp_envelopes=np.zeros((int(signals.shape[0]/frame_size)+1,signals.shape[1]))
        for i in range(signals.shape[1]):
            signal = signals[:,i]
            print(signal.shape)
            amp_envelopes[:,i]=Extract_AE(signal,frame_size)
    return(amp_envelopes)
            

def set_volume_range(signal,range):
    signal = signal/(max(abs(signal)))
    signal = signal * range
    return(signal)
    

if __name__=="__main__":
    import soundfile as sf
    length = 0.750  # seconds
    f_sample = 44100  # Hz

    f_ref = 125  # Hz, The frequency for max amplitude

    f_min = 100  # Hz
    f_max = 4000  # hz
    SNR=1
    amplification_factor=1
	# load the original audio files
    samplerate, signal = wavfile.read('digits\MAE_5A.wav')
    samplerate2, signal2 = wavfile.read('digits\MAE_1A.wav')
	# example: how to encapsulate the audio files as an nd numpy array andd zero pad the end if required
    signals=np.zeros((max(signal.shape[0],signal2.shape[0]),2))
    signals[0:signal.shape[0],0]=signal
    signals[0:signal2.shape[0],1]=signal2
	# add noise to all input signals
    sig = add_mixed_noise(signals, f_ref, f_min, f_max, length, f_sample,SNR,amplification_factor)
	
	#example of how to measure rhe loudness level of a signal 
    data, rate = sf.read("..\digits\MAE_1A.wav") # load audio (with shape (samples, channels))
    meter = pyln.Meter(rate) # create BS.1770 meter
    loudness = meter.integrated_loudness(data) # measure loudness
    print("loudness of clean signal: ",loudness,"DB LUFS")
    loudness_noisy = meter.integrated_loudness(sig)
    print("loudness noisy signal: ",loudness_noisy,"DB LUFS")

    #print("Time signal: ", sig)
	
    # plot and import the final result
    x1 = sig[:,1] * (2**15 - 1)
    x2 = sig[:,0] * (2**15 - 1)
    plt.plot(x1)
    plt.show()
    plt.plot(x2)
    plt.show()

    wavfile.write("pinkNoise1.wav", f_sample, np.array(x1, dtype=np.int16))
    wavfile.write("pinkNoise5.wav", f_sample, np.array(x2, dtype=np.int16))
	