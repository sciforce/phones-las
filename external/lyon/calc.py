import os
import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer

from .utils import design_lyon_filters, epsilon_from_tau, set_gain


class LyonCalc:
    def __init__(self, base_path=None):
        base_path = base_path or os.path.dirname(__file__)
        self._lyon_lib = ctypes.CDLL(os.path.join(base_path, 'liblyon.so'))

        self._lyon_lib.soscascade.argtypes = [
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # signal
            ctypes.c_int,  # nSamples
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # coeffs
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # state
            ctypes.c_int,  # nChannels
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")  # outputData
        ]

        self._lyon_lib.agc.argtypes = [
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # inputData
            ctypes.c_int,  # nChannels
            ctypes.c_int,  # nSamples
            ctypes.c_int,  # nStages
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # agcParams
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # state
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")  # outputData
        ]

        self._lyon_lib.sosfilters.argtypes = [
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # inputData
            ctypes.c_int,  # nInputChannels
            ctypes.c_int,  # nSamples
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # coeffs
            ctypes.c_int,  # nFilterChannels
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # state
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # outputData
            ctypes.c_int  # nOutputChannels
        ]

    def soscascade(self, signal, coeffs, state=None):
        """
        Applies soscascade.
        @parameter signal Waveform: mono, double, ndarray of N samples
        @parameter coeffs Filters: double ndarray of shape [5 x C]
        @parameter state Optional state: double ndarray of shape [2 x C]
        @returns tuple(output of shape [N x C], state)
        """
        n_samples = int(signal.size)
        n_channels = int(coeffs.shape[1])
        if state is None:
            state = np.zeros((2, n_channels), dtype=np.double)
        out = np.zeros((n_samples, n_channels), dtype=np.double)
        res = self._lyon_lib.soscascade(
            signal, n_samples, coeffs, state, n_channels, out)
        if res != 0:
            raise ValueError('Failed to apply soscascade!')
        return out, state

    def agc(self, input_data, agc_params, state=None):
        """
        Applies agc.
        @parameter input_data Input:  double ndarray of shape [N x C]
        @parameter agc_params Targets and epsilons: double ndarray of shape [S x 2]
        @parameter state Optional state: double ndarray of shape [S x C]
        @returns tuple(output of shape [N x C], state)
        """
        n_samples, n_channels = int(input_data.shape[0]), int(input_data.shape[1])
        n_stages = int(agc_params.shape[0])
        if state is None:
            state = np.zeros((n_stages, n_channels), dtype=np.double)
        out = np.zeros((n_samples, n_channels), dtype=np.double)
        res = self._lyon_lib.agc(
            input_data, n_channels, n_samples, n_stages, agc_params, state, out)
        if res != 0:
            raise ValueError('Failed to apply agc!')
        return out, state

    def sosfilters(self, input_data, coeffs, state=None):
        """
        Applies sosfilters.
        @parameter input_data Input:  double ndarray of shape [N x C]
        @parameter coeffs Filters: double ndarray of shape [5 x C]
        @parameter state Optional state: double ndarray of shape [2 x C]
        @returns tuple(output of shape [N x C], state)
        """
        n_samples, n_input_channels = int(input_data.shape[0]), int(input_data.shape[1])
        n_filter_channels = int(coeffs.shape[1])
        n_output_channels = max(n_filter_channels, n_input_channels)
        if state is None:
            state = np.zeros((2, n_output_channels), dtype=np.double)
        out = np.zeros((n_samples, n_output_channels), dtype=np.double)
        res = self._lyon_lib.sosfilters(
            input_data, n_input_channels, n_samples, coeffs, n_filter_channels,
            state, out, n_output_channels
        )
        if res != 0:
            raise ValueError('Failed to apply sosfilters!')
        return out, state

    def lyon_passive_ear(self, signal, sample_rate=16000, decimation_factor=1,
                         ear_q=8, step_factor=None, differ=True, agc=True, tau_factor=3):
        """
        Auditory nerve response using Lyon's model.
        See AuditoryToolbox for more details.
        Code below is a translation from MATLAB code.
        @parameter signal Waveform: mono, ndarray of doubles
        @parameter sample_rate Waveform sample rate, default: 16000
        @parameter decimation_factor How much to decimate model output. Default: 1
        @parameter ear_q Ear quality. Smaller values mean broader filters. Default: 8
        @parameter step_factor Filter stepping factor. Defaults to ear_q / 32 (25% overlap)
        @parameter differ Channel difference: improves model's freq response. Default: True
        @parameter agc Whether to use AGC for neural model adaptation. Default: True
        @parameter tau_factor Reduces antialiasing in filter decimation. Default: 3
        @returns ndarray of shape [N / decimation_factor, channels]
        """
        step_factor = step_factor or (ear_q / 32)
        ear_filters, _ = design_lyon_filters(sample_rate, ear_q, step_factor)
        n_samples = signal.size

        nOutputSamples = int(np.floor(n_samples / decimation_factor))
        nChannels = int(ear_filters.shape[1])

        sosOutput = np.zeros((decimation_factor, nChannels), dtype=np.double)
        sosState = np.zeros((2, nChannels), dtype=np.double)
        agcState = np.zeros((4, nChannels), dtype=np.double)
        y = np.zeros((nOutputSamples, nChannels), dtype=np.double)

        decEps = epsilon_from_tau(decimation_factor/sample_rate*tau_factor, sample_rate)
        decState = np.zeros((2, nChannels), dtype=np.double)
        _coeffs = np.array([0, 0, 1, -2*(1-decEps), (1-decEps)**2], dtype=np.double)
        decFilt = set_gain(_coeffs, 1, 0, sample_rate)

        epses = [epsilon_from_tau(x, sample_rate) for x in [.64, .16, .04, .01]]
        tars = [.0032, .0016, .0008, .0004]

        for i in range(nOutputSamples):
            window = signal[i*decimation_factor:(i + 1)*decimation_factor]
            sosOutput, sosState = self.soscascade(window, ear_filters, sosState)
            output = np.clip(sosOutput, 0, None)  # Half Wave Rectify
            output[0, 0] = 0  # Test Hack to make inversion easier.
            output[0, 1] = 0
            if agc:
                agc_params = np.array(list(zip(tars, epses)), dtype=np.double)
                output, agcState = self.agc(output, agc_params, agcState)
            if differ:
                output = np.concatenate([output[:, 0:1], output[:, :-1] - output[:, 1:]], axis=1)
                output = np.clip(output, 0, None)
            if decimation_factor > 1:
                output, decState = self.sosfilters(output, decFilt[:, np.newaxis], decState)
            y[i, :] = output[-1, :]

        return y[:, 2:]
