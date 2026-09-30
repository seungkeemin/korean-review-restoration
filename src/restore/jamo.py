"""Hangul syllable <-> (initial, medial, final) jamo."""

BASE_CODE = 0xAC00
CHOSUNG = 21 * 28
JUNGSUNG = 28

CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
                'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
                 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
                 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


def decompose_syllable(ch):
    code = ord(ch) - BASE_CODE
    if code < 0 or code > (0xD7A3 - 0xAC00):
        return None
    cho = code // CHOSUNG
    jung = (code % CHOSUNG) // JUNGSUNG
    jong = (code % CHOSUNG) % JUNGSUNG
    return (CHOSUNG_LIST[cho], JUNGSUNG_LIST[jung], JONGSUNG_LIST[jong])


def compose_syllable(cho, jung, jong):
    if cho not in CHOSUNG_LIST or jung not in JUNGSUNG_LIST or jong not in JONGSUNG_LIST:
        return None
    code = (BASE_CODE + CHOSUNG_LIST.index(cho) * CHOSUNG
            + JUNGSUNG_LIST.index(jung) * JUNGSUNG + JONGSUNG_LIST.index(jong))
    return chr(code)


def to_jamo_list(text):
    """String -> [(cho, jung, jong)], with non-Hangul characters kept as (ch, '', '')."""
    out = []
    for ch in text:
        d = decompose_syllable(ch)
        out.append((ch, '', '') if d is None else d)
    return out


def from_jamo_list(jamo_list):
    out = []
    for cho, jung, jong in jamo_list:
        syl = compose_syllable(cho, jung, jong)
        if syl:
            out.append(syl)
        else:
            out.append(cho if jung == '' and jong == '' else cho + jung + jong)
    return "".join(out)
