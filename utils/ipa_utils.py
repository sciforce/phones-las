from utils.espeakng import ESpeakNG
import re
import itertools
import pandas as pd
import numpy as np

from utils.vocab_utils import SOS, EOS, UNK, SOS_ID, EOS_ID, UNK_ID

__all__ = [
    'IPAError',
    'get_ipa',
    'load_binf2phone',
    'ipa2binf',
    'get_mapping'
]

engine = ESpeakNG()

class IPAError(ValueError):
    pass

DIACRITICS_LIST = r'̆|\.|\||‖|↗|↘|\d|-'
STRESS_DIACRITICS = r'ˈ|ˌ'
GEMINATION_DIACRITICS = r'ː|ˑ'

SEP = ','

VOWELS = ['i', 'ĩ', 'y', 'ɨ', 'ʉ', 'ɯ', 'u', 'ü', 'ũ', 'ɪ', 'ʏ', 'ɪ̈', 'ᵻ', 'ʊ̈', 'ɯ̽', 'ʊ', 'e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o', 'e̞', 'ø̞', 'ə', 'ɚ', 'ɤ̞', 'ɔ̝', 'o̞', 'ɛ', 'ε', 'œ', 'ɜ', 'ɝ', 'ɞ', 'ʌ', 'ɔ', 'æ', 'ɐ', 'a', 'ã', 'A', 'ɶ', 'ä', 'ɑ', 'ɒ']
CONSONANTS = ['m̥', 'm', 'ɱ', 'n̥', 'n', 'ɳ̊', 'ɳ', 'ɲ̊', 'ɲ', 'ŋ̊', 'ŋ', 'ɴ', 'p', 'b', 'p̪', 'b̪', 't̪', 'd̪', 't', 'd', 'ʈ', 'ɖ', 'c', 'ɟ', 'k', 'ɡ', 'g', 'q', 'ɢ', 'ʡ', 'ʔ', 's', 'z', 'ʃ', 'ʒ', 'ʂ', 'ʐ', 'ɕ', 'ʑ', 'ɸ', 'β', 'f', 'v', 'θ', 'ð', 'θ̱', 'ð̠', 'ɹ̠̊˔', 'ɹ̠˔', 'ç', 'ʝ', 'x', 'ɣ', 'χ', 'ʁ̝', 'ʁ', 'ħ', 'ʕ̝', 'ʕ', 'ʜ', 'ʢ̝', 'ʢ', 'h', 'ɦ', 'β̞', 'ʋ', 'ð̞', 'ɹ', 'ɻ', 'j̊', 'j', 'ɰ', 'ʁ̞', 'ʕ̞', 'ʢ̞', 'ʙ', 'r̥', 'r', 'ɽ͡r', 'ʀ̥', 'ʀ', 'ᴙ', 'ɬ', 'ɮ', 'ꞎ', 'ʎ̝̊', 'ʟ̝̊', 'ʟ̝', 'l̥', 'l', 'ɭ', 'ʎ', 'ʟ', 'ɫ', 'ɺ', 'ɺ̢', 'ʎ̯', 'ʟ̆', 'ʍ', 'w̥', 'w', 'ɥ', 'ɧ', 'k͜p', 'ɡ͡b', 'ŋ͡m', 'p̪͡f', 'b̪͡v', 't͡s', 't͜s', 'ʦ', 'ts', 'd͡z', 'd͜z', 'ʣ', 'dz', 't̪͡s̪', 't͡s̪', 't̟͡s̟', 't͡s̟', 'd̪͡z̪', 'd͡z̪', 'd̟͡z̟', 'd͡z̟', 't͡θ', 't͜θ', 't̪θ', 't̪͡θ', 't̟͡θ', 'd̪ð', 'd͡ð', 'd͜ð', 'd̪͡ð', 'd̟͡ð', 't͡ʃ', 't͜ʃ', 'ʧ', 't̠ʲʃ', 'tʃ', 'd͡ʒ', 'd͜ʒ', 'ʤ', 'd̠ʲʒ', 'd̠ʒ', 'dʒ', 't͡ɕ', 't͜ɕ', 'ʨ', 't̠ɕ', 'tɕ', 'd͡ʑ', 'd͜ʑ', 'ʥ', 'd̠ʑ', 'dʑ', 'ʈ͡ʂ', 't͡ʂ', 'tʂ', 'ɖ͡ʐ', 'd͡ʐ', 'dʐ', 't͡ɬ', 't͜ɬ', 'tɬ', 'ƛ', 'd͡ɮ', 'dɮ', 'λ', 'c͡ç', 'c͜ç', 'ɟ͡ʝ', 'c͡ʎ̝̥', 'k͡x', 'ɡ͡ɣ', 'k͡ʟ̝̊', 'ɡ͡ʟ̝', 'q͡χ', 'ɢ͡ʁ', 'ɢ͜ʁ', 'pɸ', 'p͡ɸ', 'bβ', 'b͡β', 'p̪f', 'b̪v', 'ʘ', 'ǀ', 'ǃ', 'ǂ', 'ǁ', '‼', 'ɓ̥', 'ɓ', 'ɗ̥', 'ɗ', 'ʄ̊', 'ʄ', 'ɠ', 'ʛ']
VOWELS_CORE = [v for v in VOWELS if len(v) == 1]

# Consonants used at old IPA service:
# consonants = ['m', 'n', 'ɴ', 's', 'z', 'ʃ', 'ʒ', 'ʂ', 'ʐ', 'ɕ', 'ʑ', 'ɸ', 'β', 'f', 'v', 'f', 'θ', 'ð',
#                           'ç', 'ʝ', 'x', 'ɣ', 'χ', 'ʁ', 'ħ', 'ʕ', 'ʜ', 'ʢ', 'h', 'ɦ', 'ʋ', 'ɹ', 'ɻ', 'j', 'ɰ', 'ⱱ',
#                           'ɾ', 'ɽ', 'ʙ', 'r', 'ʀ', 'ᴙ', 'ɬ', 'ɮ', 'ꞎ', 'l', 'ɭ', 'ʎ', 'ʟ', 'ɫ', 'ɺ']

voices = [l['language'] for l in engine.voices]

def _japanese_preprocessing(text):
    #TODO: replace Kanji by Hiragana/Katakana
    return text

def _preprocessing(text, language):
    if language == 'ja':
        text = _japanese_preprocessing(text)
    elif language == 'fr':
        text = re.sub(r'à|À', 'a', text)
    elif language == 'ru':
        text = re.sub(u'\u0301', '', text)
    elif language == 'ur':
        text = re.sub(u'\u064E|\u0650|\u064F|\u0652', '', text)
    #remove punctuation (otherwise eSpeak will not return spaces)
    text = re.sub(r'([^\w\s])', '', text)
    return text

def _postprocessing(ipa, language, remove_all_diacritics=False,
    split_all_diphthongs=False):
    lang_markers_pattern = r'(\([^)]+\))'
    if language != 'ja':
        # remove language switch markers
        ipa = re.sub(lang_markers_pattern, '', ipa)
    elif re.search(lang_markers_pattern,ipa) is not None:
        # In Japanese, language switch markers indicate
        # presence of character names instead of phonemes.
        # Such transcription is useless.
        raise IPAError(ipa)
    if remove_all_diacritics:
        # remove diacritics
        ipa = ''.join(x for x in ipa if x.isalnum())
    else:
        ipa = re.sub(DIACRITICS_LIST, '', ipa)
    ipa = re.sub(r'([\r\n])', ' ', ipa)
    # split by phonenes, keeping spaces
    ipa = [p for word in ipa.split(' ') for p in itertools.chain(word.split('_'), ' ') if p != '']
    ipa = ipa[:-1]
    ipa = _postprocess_double_consonants(ipa)
    ipa = _postprocess_double_vowels(ipa)
    return _postprocess_by_languages(ipa, language, split_all_diphthongs)

def _postprocess_double_consonants(text):
    out_text = []
    for i, _ in enumerate(text):
        del_phone = False
        for cons in CONSONANTS:
            text[i] = re.sub('({cons}\w?){cons}\w?'.format(cons=cons), r'\1ː', text[i])
            if cons in text[i] and i > 0 and out_text[-1] == text[i]:
                out_text[-1] += 'ː'
                del_phone = True
        if not del_phone:
            out_text.append(text[i])
    return out_text

def _postprocess_double_vowels(text):
    for i, _ in enumerate(text):
        for vow in VOWELS:
            text[i] = re.sub('([ˈ|ˌ]?{vow}\w?){vow}\w?'.format(vow=vow), r'\1ː', text[i])
    return text

def _process_diphthongs(text, split_all_diphthongs=False):
    # split triphthongs
    text = re.sub(r'([{vow}]\w*)([{vow}]\w*)([{vow}])'.format(vow=''.join(VOWELS_CORE)),
        r'\1\2{sep}\3'.format(sep=SEP), text)
    if split_all_diphthongs:
        text = re.sub(r'([{vow}]\w*)([{vow}])'.format(vow=''.join(VOWELS_CORE)),
            r'\1{sep}\2'.format(sep=SEP), text)
    else:
        text = re.sub(r'([{vow}]\w*[ːˑ])([{vow}])'.format(vow=''.join(VOWELS_CORE)),
            r'\1{sep}\2'.format(sep=SEP), text)
    return text


def _postprocess_by_languages(text, language, split_all_diphthongs):
    text = SEP.join(text)
    text = re.sub(r'(\w+ː)ː', r'\1', text)
    if 'vi' in language:
        text = re.sub('kh', 'k̚ʷ', text)
    if 'en' in language:
        text = re.sub('əl', 'l', text)
        text = re.sub('(\w+)ɹ', r'\1˞', text)
    if language == 'de':
        text = re.sub('pf', 'p̪f', text)
    if language == 'la':
        text = re.sub('kːʰ', 'kʰː', text)
    if language == 'ga':
        text = re.sub(r'({sep}\s{sep})ʲ{sep}'.format(sep=SEP),
            r'ʲ\1', text)
    if language == 'ro':
        text = re.sub(r'(\w+ʲ)\w*ʲ', r'\1', text)
    if language == 'is':
        text = re.sub(r'([nlrmɲŋ])#', r'\1̥', text)
        text = re.sub('tl', 't{sep}l'.format(sep=SEP), text)
    if language == 'et' or language == 'mk':
        text = re.sub(r'(\w)\^', r'\1ʲ', text)
    if language == 'da':
        #TODO: Parse Stød properly (https://en.wikipedia.org/wiki/Stød)
        text = re.sub(r'[ʔ\?]([yeœʌeiouɑa])', r'\1', text)
    if language == 'lv':
        text = re.sub('{sep}[>}}]'.format(sep=SEP), '', text)
        text = re.sub('^[>}}]{sep}'.format(sep=SEP), '', text)
        text = re.sub('`', '', text)
    if language == 'sv':
        text = re.sub('sx', 'ɧ', text)
    if language == 'ru':
        text = re.sub('u"', 'ü', text)
        text = re.sub(r'(\w){sep}ɪ\^'.format(sep=SEP), r'\1ʲ', text)
        # Separate ju, ja
        text = re.sub(r'([ˈˌ]?)(j)([ua])', r'\2{sep}\1\3'.format(sep=SEP), text)
    if language == 'zh':
        text = re.sub(r'(\w+)ɜ', r'\1', text)
        text = re.sub('onɡ', 'o{sep}n{sep}ɡ'.format(sep=SEP), text)
        text = re.sub(r'(\w)h', r'\1ʰ', text)
        text = re.sub('ər', 'ɚ', text)
        text = re.sub('{sep}ʲ'.format(sep=SEP), '', text)
        text = re.sub('^ʲ{sep}'.format(sep=SEP), '', text)
    if language == 'ko':
        text = re.sub(r'(\w)h', r'\1ʰ', text)
    if language == 'ja':
        # Delete short u (‘ɯ’) between voiceless consonants or at the end of the word.
        sub_reg_template = ('((?:m̥|n̥|ɳ̊|ɲ̊|ŋ̊|p|p̪|t̪|t|ʈ|c|k|'
                            'q|ʡ|ʔ|[^t]{sep}?s|^s|ʃ|ʂ|[^t]ɕ|^ɕ|ɸ|f|f|θ|θ̱|ɹ̠̊˔|ç|'
                            'x|χ|ħ|ʜ|h|j̊|ɾ̥|r̥|ʀ̥|ɬ|ꞎ|ʎ̝̊|ʟ̝̊|l̥|'
                            'ʍ|w̥|ɧ|k͜p|p̪͡f|t͡s|t͜s|ʦ|t̪͡s̪|t͡s̪|t̟͡s̟|'
                            't͡s̟|t͡θ|t͜θ|t̪͡θ|t̟͡θ|t͡ʃ|t͜ʃ|ʧ|t̠ʲʃ|tʃ|'
                            't͡ɕ|t͜ɕ|ʨ|ʈ͡ʂ|t͡ʂ|tʂ|t͡ɬ|t͜ɬ|tɬ|ƛ|c͡ç|'
                            'c͜ç|c͡ʎ̝̥|k͡x|k͡ʟ̝̊|q͡χ|ʘ|ǀ|ǃ|ǂ|ǁ|‼|ɓ̥|ɗ̥|ʄ̊){sep})((?:ɯᵝ|ɯ)(?:{sep}|$))('
                            'm̥|n̥|ɳ̊|ɲ̊|ŋ̊|p|p̪|t̪|t{sep}?[^s|^ɕ]|t$|ʈ|c|k|'
                            'q|ʡ|ʔ|s|ʃ|ʂ|ɕ|ɸ|f|f|θ|θ̱|ɹ̠̊˔|ç|'
                            'x|χ|ħ|ʜ|h|j̊|ɾ̥|r̥|ʀ̥|ɬ|ꞎ|ʎ̝̊|ʟ̝̊|l̥|'
                            'ʍ|w̥|ɧ|k͜p|p̪͡f|t͡s|t͜s|ʦ|t̪͡s̪|t͡s̪|t̟͡s̟|'
                            't͡s̟|t͡θ|t͜θ|t̪͡θ|t̟͡θ|t͡ʃ|t͜ʃ|ʧ|t̠ʲʃ|tʃ|'
                            't͡ɕ|t͜ɕ|ʨ|ʈ͡ʂ|t͡ʂ|tʂ|t͡ɬ|t͜ɬ|tɬ|ƛ|c͡ç|'
                            'c͜ç|c͡ʎ̝̥|k͡x|k͡ʟ̝̊|q͡χ|ʘ|ǀ|ǃ|ǂ|ǁ|‼|ɓ̥|ɗ̥|ʄ̊)')
        text = re.sub(sub_reg_template.format(sep=SEP), r"\1\3", text)
        # Delete short i after 'ɕ'
        text = re.sub('([^t]ɕ|[^t]ɕˈ|^ɕ){sep}i()'.format(sep=SEP), r"\1\2", text)
        # Replaced on long voice
        text = re.sub('(a|k{sep}a|s{sep}a|t{sep}a|n{sep}a|h{sep}a|'
                      'm{sep}a|j{sep}a|r{sep}a|w{sep}a|g{sep}a|'
                      'd{sep}z{sep}a|d{sep}a|b{sep}a|p{sep}a){sep}a()'.format(sep=SEP),
                      r"\1ː\2", text)
        text = re.sub('(i|k{sep}i|ɕ{sep}i|t{sep}ɕ{sep}i|n{sep}i|h{sep}i|m{sep}i|'
                      'r{sep}i|w{sep}i|g{sep}i|ʑ{sep}i|d{sep}ʑ{sep}i|b{sep}i'
                      '|p{sep}i){sep}i()'.format(sep=SEP), r"\1ː\2", text)
        text = re.sub('(ɯ|k{sep}ɯ|s{sep}ɯ|t{sep}s{sep}ɯ|n{sep}ɯ|f{sep}ɯ|m{sep}ɯ|'
                      'j{sep}ɯ|r{sep}ɯ|g{sep}ɯ|z{sep}ɯ|d{sep}z{sep}ɯ|b{sep}ɯ|'
                      'p{sep}ɯ){sep}ɯ()'.format(sep=SEP), r"\1ː\2", text)
        text = re.sub('(e|k{sep}e|s{sep}e|t{sep}e|n{sep}e|h{sep}e|m{sep}e|'
                      'r{sep}e|w{sep}e|g{sep}e|z{sep}e|d{sep}e|b{sep}e|'
                      'p{sep}e){sep}e()'.format(sep=SEP), r"\1ː\2", text)
        text = re.sub('(o|k{sep}o|s{sep}o|t{sep}o|n{sep}o|h{sep}o|'
                      'm{sep}o|j{sep}o|r{sep}o|w{sep}o|g{sep}o|'
                      'z{sep}o|d{sep}o|b{sep}o|p{sep}o)({sep}[o|'
                      'ɯ])()'.format(sep=SEP), r"\1ː\3", text)

        # Delete short u (‘ɯ’) at the end of the word.
        text = re.sub(r'(\w)({sep}(?:ɯᵝ|ɯ))($|{sep}\s)'.format(sep=SEP), r"\1\3", text)

    text = _process_diphthongs(text, split_all_diphthongs=split_all_diphthongs)

    text = text.split(SEP)

    # Remove standalone sterss (occurs in Welsh, Latin)
    text = list(filter(lambda x: x != 'ˈ', text))

    return text

def get_ipa(text, language, remove_all_diacritics=False,
    split_all_diphthongs=False):
    engine.voice = language
    text = _preprocessing(text, engine.voice)
    # get ipa with '_' as phonemes separator
    process_by_words = False
    if engine.voice == 'ko':
        process_by_words = True
    if not process_by_words:
        ipa = engine.g2p(text, ipa=1)
    else:
        text = text.split(' ')
        ipa = ' '.join(engine.g2p(word, ipa=1) for word in text)
    if ipa.startswith('Error:'):
        raise IPAError(ipa)
    return _postprocessing(ipa, engine.voice, remove_all_diacritics,
        split_all_diphthongs)

def get_mapping(mapping_path, vocab_path):
    with open(mapping_path, 'r') as fid:
        mapping_lines = fid.read().strip().replace('sil', 'h#').split('\n')
    with open(vocab_path, 'r') as fid:
        vocab = fid.read().strip().split('\n')
    mapping = dict()
    new_vocab = set()
    for line in mapping_lines:
        phones = line.split('\t')
        if len(phones) < 3:
            mapping[phones[0]] = None
        else:
            mapping[phones[0]] = phones[-1]
            new_vocab.update([phones[-1]])
    new_vocab = [UNK, SOS, EOS] + sorted(new_vocab)
    int_mapping = [0, 1, 2]
    for phone in vocab:
        if mapping[phone] is not None:
            int_mapping.append(new_vocab.index(mapping[phone]))
        else:
            int_mapping.append(-1)
    return new_vocab, int_mapping

def load_binf2phone(filename, vocab_list=None):
    binf2phone = pd.read_csv(filename, index_col=0)
    binf2phone.insert(UNK_ID, UNK, 1)
    binf2phone.insert(SOS_ID, SOS, 0)
    binf2phone.insert(EOS_ID, EOS, 0)
    
    bottom_df = pd.DataFrame(np.zeros([2, binf2phone.shape[1]]),
                             columns=binf2phone.columns, index=[SOS, EOS])
    binf2phone = pd.concat((binf2phone, bottom_df))
    binf2phone.loc[binf2phone.index==SOS, SOS] = 1
    binf2phone.loc[binf2phone.index==EOS, EOS] = 1
    binf2phone.loc[binf2phone.index.isin([SOS, EOS]), UNK] = 1
    if vocab_list is not None:
        # Leave only phonemes from the vocabluary
        binf2phone = binf2phone[vocab_list]
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