"""Check AuditoryToolbox for more details on function in these module."""
import numpy as np


def second_order_filter(f, q, fs):
    """
    Second order filter.
    @returns [1, b1, b2]
    """
    cft = f / fs
    rho = np.exp(- np.pi * cft / q)
    theta = 2 * np.pi * cft * np.sqrt(1 - 1.0 / (4*np.square(q)))
    filts = np.array([np.ones_like(rho), -2 * rho * np.cos(theta), np.square(rho)])
    return filts


def freq_resp(filter, f, fs):
    cf = np.exp(1j * 2 * np.pi * f / fs)
    denom = filter[4] + filter[3]*cf + cf**2
    mag = (filter[2] + filter[1]*cf + filter[0]*np.square(cf)) / denom
    mag = 20*np.log10(np.abs(mag))
    return mag


def set_gain(filter, desired, f, fs):
    """
    Set the gain of a filter (1x5 vector) to any desired gain at any desired frequency (f).
    """
    old_gain = 10 ** (freq_resp(filter, f, fs) / 20)
    filter[:3] = filter[:3] * desired / old_gain
    return filter


def design_lyon_filters(fs, ear_q=8, step_factor=None):
    step_factor = step_factor or (ear_q / 32)
    Eb = 1000.0
    EarZeroOffset = 1.5
    EarSharpness = 5.0
    EarPremphCorner = 300

    # Find top frequency, allowing space for first cascade filter.
    topf = fs/2.0
    topf_neg_delta = (np.sqrt(topf**2+Eb**2)/ear_q*step_factor*EarZeroOffset)
    topf = topf - topf_neg_delta + np.sqrt(topf**2+Eb**2)/ear_q*step_factor

    # Find place where CascadePoleQ < .5
    lowf = Eb/np.sqrt(4*ear_q**2-1)
    _log_val_low = np.log(lowf + np.sqrt(lowf**2 + Eb**2))
    _log_val_top = np.log(topf + np.sqrt(Eb**2 + topf**2))
    NumberOfChannels = np.floor(
        (ear_q*(-_log_val_low + _log_val_top))/step_factor
    )

    # Now make an array of CenterFreqs..... This expression was derived by
    # Mathematica by integrating 1/EarBandwidth(cf) and solving for f as a
    # function of channel number.
    cn = np.arange(1, NumberOfChannels + 1)
    denom = np.exp((cn*step_factor)/ear_q)
    nom_add = topf + np.sqrt(Eb**2 + topf**2)
    center_freqs = (-((np.exp((cn*step_factor)/ear_q)*Eb**2) / nom_add) + nom_add / denom) / 2.0

    # OK, now we can figure out the parameters of each stage filter.
    EarBandwidth = np.sqrt(np.square(center_freqs) + Eb**2) / ear_q
    CascadeZeroCF = center_freqs + EarBandwidth * step_factor * EarZeroOffset
    CascadeZeroQ = EarSharpness * CascadeZeroCF / EarBandwidth
    CascadePoleCF = center_freqs
    CascadePoleQ = center_freqs / EarBandwidth

    # Now lets find some filters.... first the zeros then the poles
    zerofilts = second_order_filter(CascadeZeroCF, CascadeZeroQ, fs)
    polefilts = second_order_filter(CascadePoleCF, CascadePoleQ, fs)
    filters = np.vstack([zerofilts, polefilts[1:, :]])

    # Now we can set the DC gain of each stage.
    dcgain = np.zeros((int(NumberOfChannels),), dtype=np.double)
    dcgain[1:] = center_freqs[:-1] / center_freqs[1:]
    dcgain[0] = dcgain[1]
    for i in range(int(NumberOfChannels)):
        filters[:, i] = set_gain(filters[:, i], dcgain[i], 0, fs)

    # Finally, let's design the front filters.
    front = np.zeros((5, 2), dtype=np.double)
    front_0 = np.array([0, 1, -np.exp(-2*np.pi*EarPremphCorner/fs), 0, 0], dtype=np.double)
    front[:, 0] = set_gain(front_0, 1, fs/4, fs)
    top_poles = second_order_filter(topf, CascadePoleQ[0], fs)
    front_1 = np.array([1, 0, -1, top_poles[1], top_poles[2]], dtype=np.double)
    front[:, 1] = set_gain(front_1, 1, fs/4, fs)

    # Now, put them all together.
    filters = np.hstack([front, filters])
    return filters, center_freqs


def epsilon_from_tau(tau, sample_rate):
    return 1-np.exp(-1.0/tau/sample_rate)
