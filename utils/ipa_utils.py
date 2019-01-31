from espeakng import ESpeakNG
import re
import itertools
import pandas as pd
import numpy as np

from utils.vocab_utils import SOS, EOS, UNK, SOS_ID, EOS_ID, UNK_ID


__all__ = [
    'IPAError',
    'get_ipa',
    'load_binf2phone',
    'ipa2binf'
]

SPE_NAMES = ['vocalic',
 'consonantal',
 'high',
 'back',
 'low',
 'anterior',
 'coronal',
 'round',
 'tense',
 'voiced',
 'continuant',
 'nasal',
 'strident',
 'silence',
 'vowel_group',
 'fricative_group',
 'stop_manner']

DIACRITICS_LIST = r'ˈ|ˌ|ː|ˑ|̆|\.|\||‖|↗|↘|\d'

class IPAError(ValueError):
    pass


def _postprocessing(ipa, remove_all_diacritics=False):
    # remove language switch markers
    ipa = re.sub(r'(\([^)]+\))', '', ipa)
    if remove_all_diacritics:
        # remove diacritics
        ipa = ''.join(x for x in ipa if x.isalnum())
    else:
        ipa = re.sub(DIACRITICS_LIST, '', ipa)
    ipa = re.sub(r'([\r\n])', ' ', ipa)
    # split by phonenes, keeping spaces
    ipa = [p for word in ipa.split(' ') for p in itertools.chain(word.split('_'), ' ') if p != '']
    ipa = ipa[:-1]
    return ipa

def _preprocessing(text):
    #remove punctuation (otherwise eSpeak will not return spaces)
    text = re.sub(r'([^\w\s])', '', text)
    return text

def get_ipa(text, language, remove_all_diacritics=False):
    engine = ESpeakNG()
    engine.voice = language.lower()
    text = _preprocessing(text)
    # get ipa with '_' as phonemes separator
    ipa = engine.g2p(text, ipa=1)
    if ipa.startswith('Error:'):
        raise IPAError(ipa)
    return _postprocessing(ipa, remove_all_diacritics)

def load_binf2phone(filename, vocab_list=None):
    binf2phone = pd.read_csv(filename, index_col=0)
    if vocab_list is not None:
        # Leave only phonemes from the vocabluary
        new_cols = [col for col in binf2phone.columns if col in vocab_list]
        binf2phone = binf2phone[new_cols]
    binf2phone.insert(UNK_ID, UNK, 1)
    binf2phone.insert(SOS_ID, SOS, 0)
    binf2phone.insert(EOS_ID, EOS, 0)
    
    bottom_df = pd.DataFrame(np.zeros([2, binf2phone.shape[1]]),
                             columns=binf2phone.columns, index=[SOS, EOS])
    binf2phone = pd.concat((binf2phone, bottom_df))
    binf2phone.loc[binf2phone.index==SOS, SOS] = 1
    binf2phone.loc[binf2phone.index==EOS, EOS] = 1
    binf2phone.loc[binf2phone.index.isin([SOS, EOS]), UNK] = 1
    return binf2phone

def ipa2binf(ipa, binf2phone, try_merge_diphtongs=False):
    binf = np.empty((len(ipa), len(binf2phone.index)), np.float32)
    for k, phone in enumerate(ipa):
        if phone in binf2phone.columns:
            binf_vec = binf2phone[phone].values
        elif len(phone) > 1 and try_merge_diphtongs:
            try:
                binf_vec = np.zeros((len(binf2phone.index)), np.int)
                for p in phone:
                    binf_vec = np.logical_or(binf_vec, binf2phone[p].values).astype(np.float32)
            except KeyError:
                raise IPAError(phone)
        else:
            raise IPAError(phone)
        binf[k, :] = binf_vec
    return binf