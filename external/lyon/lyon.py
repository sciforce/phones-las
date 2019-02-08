import os
import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer


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
        state = state or np.zeros((2, n_channels), dtype=np.double)
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
        state = state or np.zeros((n_stages, n_channels), dtype=np.double)
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
        state = state or np.zeros((2, n_output_channels), dtype=np.double)
        out = np.zeros((n_samples, n_output_channels), dtype=np.double)
        res = self._lyon_lib.sosfilters(
            input_data, n_input_channels, n_samples, coeffs, n_filter_channels,
            state, out, n_output_channels
        )
        if res != 0:
            raise ValueError('Failed to apply sosfilters!')
        return out, state
