#!/usr/bin/env python3
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-06-20

"""
Unicode and UTF are designed for the rendering of fonts while reducing the memory and cpu requirements
Some of the decisions made there do not reflect how a human linguistic looks at the characters.
In particular:
 1. a consonant always comes with an implied vowel.
 2. A consonant-without-vowel is actually consonant-with-vowel + virama ; virama removes the vowel

So, in summary Unicode/UTF's consonant starts with implied vowel,
 then followed by a subtraction or modification of vowel is performed.
Naturally, we start with a base consonant without any vowel, followed addition of
 zero or one or two vowels.


## Known issues:
 case1:
  implicit vowel followed by explicit vowel -> implicit vowel is lost during code-decode
  example गए -> गे after encoding and decoding
"""

import argparse
import sys
from typing import List, Set, Union, Dict, Optional, Iterator

# Note: Unicode ranges are reserved for private use
#   Basic Multilingual plane :: U+E000 - U+F8FF
#   planes 15 :: U+F0000 – U+FFFFD
#   Plane 16 :: U+100000 – U+10FFFD
# we will use some for our private use

# full-vowel without attached to any consonant, inside a word
internal_full_vowel = 0xE000


class Lang:

    def __init__(self, name: str, vowels: Set[int], consonants: Set[int]):
        self.name = name
        self.vowels = vowels
        self.consonants = consonants
        assert isinstance(vowels, set)
        assert isinstance(consonants, set)
        confused = vowels & consonants
        assert not confused, [chr(c) for c in confused]
        self.spaces = {ord(' '), ord('\t'), ord('\n')}

    def is_vowel(self, code):
        return code in self.vowels

    def is_consonant(self, code):
        return code in self.consonants

    def is_space(self, code):
        return code in self.spaces

    @classmethod
    def maybe_ord(cls, obj):
        if obj is None:
            return None
        if isinstance(obj, int):
            return obj
        elif isinstance(obj, str):
            return ord(obj)
        elif isinstance(obj, dict):
            return {cls.maybe_ord(k): cls.maybe_ord(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return list(map(cls.maybe_ord, obj))
        elif isinstance(obj, set):
            return set(map(cls.maybe_ord, obj))
        else:
            raise Exception(f'{obj} of type {type(obj)} is not compatible')

    def codes_to_str(self, codes: List[int]) -> str:
        line = ''.join(chr(c) for c in codes if c is not None)
        # return line.encode('utf8').decode('utf8')
        return line


CodePt = Union[int, str]


class AbugidaLang(Lang):

    def __init__(self, name: str, vowels: Dict[CodePt, Optional[CodePt]],
                 consonants: Set[CodePt],
                 impl_vowel: CodePt, virama: CodePt, semi_vowels: Set[CodePt] = None):
        """
        :param name: name of lang. ISO 3 letter code recommended
        :param vowels: map of vowel -> symbol
        :param consonants: list of consonants with implicit vowel
        :param impl_vowel: the implicit vowel in the consonant codepoint
        :param virama: virama symbol. this removes the implicit vowel from the consonants
        """
        _ord = self.maybe_ord
        vowels = _ord(vowels)
        consonants = _ord(consonants)
        assert len(vowels) == len(set(vowels.values()))  # one to one

        super().__init__(name, set(vowels.keys()), consonants)
        self.impl_vowel = _ord(impl_vowel)
        self.virama = _ord(virama)
        self.vowel_to_form = _ord(vowels)
        self.form_to_vowel = {f: v for v, f in vowels.items()}
        self.semi_vowels = _ord(semi_vowels or set())

    def is_vowel_form(self, code):
        # There is None to the implicit vowel
        return code and code in self.form_to_vowel

    def is_semi_vowel(self, code):
        return code and code in self.semi_vowels

    def is_virama(self, code):
        return code == self.virama

    def encode(self, line: str) -> str:
        codes = list(map(ord, line))
        enc = []
        i = 0
        while i < len(codes):
            if self.is_virama(codes[i]):
                pass  # skip virama; treat consonant itself as consonant+virama
            else:
                is_ending = i + 1 == len(codes) or self.is_space(codes[i + 1])
                is_beginning = i == 0 or (i > 0 and self.is_space(codes[i - 1]))
                if self.is_vowel(codes[i]):  # full vowel goes as in
                    enc.append(codes[i])
                    if not is_beginning:
                        # full vowel inside a word, that's linguistically ugly; special casing it
                        enc.append(internal_full_vowel)
                elif self.is_consonant(codes[i]):
                    enc.append(codes[i])  # copy consonant
                    # implied vowel, if there is no next or next is neither virama not vowel_form
                    if is_ending or not (self.is_vowel_form(codes[i + 1])
                                         or self.is_virama(codes[i + 1])):
                        enc.append(self.impl_vowel)  # implied vowel
                elif self.is_vowel_form(codes[i]):
                    enc.append(self.form_to_vowel[codes[i]])
                else:  # default action
                    enc.append(codes[i])  # just copy
            i += 1
        return self.codes_to_str(enc)

    def decode(self, line) -> str:
        codes = list(map(ord, line))
        dec = []
        i = 0
        while i < len(codes):
            #print(i, chr(codes[i]), [chr(c) if c else c for c in dec])
            # word ending and beginning
            is_ending = i + 1 == len(codes) or self.is_space(codes[i + 1])
            is_beginning = i == 0 or self.is_space(codes[i - 1])
            if codes[i] == internal_full_vowel:
                pass  # this shouldn't be happening. but the input data can be noisy
            elif self.is_consonant(codes[i]):
                dec.append(codes[i])  # copy and look ahead next
                if is_ending or self.is_consonant(codes[i + 1]):
                    dec.append(self.virama)  # restore virama between two consonants
            elif self.is_vowel(codes[i]):
                # look ahead, dont modify this vowel to its form if there is a marker
                if not is_ending and codes[i + 1] == internal_full_vowel:
                    dec.append(codes[i])
                    i += 1  # skip the next code point `internal_full_vowel` marker
                elif is_beginning:  # vowel in the beginning gets copied
                    dec.append(codes[i])
                else:  # something was before
                    if self.is_consonant(codes[i - 1]):
                        dec.append(self.vowel_to_form[codes[i]])  # vowel to modifier form
                    else:  # two consecutive vowels,
                        # TODO: advanced vowel modifications
                        # refer sanskrit sandhi http://www.learnsanskrit.org/references/sandhi/vowel
                        # dec.append(self.vowel_to_form[codes[i]])
                        dec.append(self.vowel_to_form[codes[i]])  # second vowel goes without changing to modifier form
            else:  # default copy
                dec.append(codes[i])
            i += 1
        return self.codes_to_str(dec)

    def encode_all(self, lines: Iterator[str]) -> Iterator[str]:
        yield from (self.encode(line.rstrip("\n")) for line in lines)

    def decode_all(self, lines: Iterator[str]) -> Iterator[str]:
        yield from (self.decode(line.rstrip("\n")) for line in lines)


class AbugidaLangLite(AbugidaLang):
    """
    Lite modification: remove virama, convert implicit to explicit vowel
    """

    def encode(self, line: str) -> str:
        codes = list(map(ord, line))
        enc = []
        i = 0
        while i < len(codes):
            if self.is_virama(codes[i]):
                pass  # skip virama; treat consonant itself as consonant+virama
            else:
                is_ending = i + 1 == len(codes) or self.is_space(codes[i + 1])
                #is_beginning = i == 0 or (i > 0 and self.is_space(codes[i - 1]))
                enc.append(codes[i])
                if self.is_consonant(codes[i]):
                    if is_ending or self.is_consonant(codes[i+1]):
                        enc.append(self.impl_vowel)
            i += 1
        return self.codes_to_str(enc)

    def decode(self, line) -> str:
        codes = list(map(ord, line))
        dec = []
        i = 0
        while i < len(codes):
            is_ending = i + 1 == len(codes) or self.is_space(codes[i + 1])
            is_beginning = i == 0 or self.is_space(codes[i - 1])
            dec.append(codes[i])  # copy
            if self.is_consonant(codes[i]): # look ahead
                if is_ending or self.is_consonant(codes[i + 1]):
                    dec.append(self.virama)  # restore virama between two consonants
                elif self.impl_vowel == codes[i+1]:
                    i += 1 # skip Implied vowel
            i += 1
        return self.codes_to_str(dec)

def get_aubugida(lang_name) -> AbugidaLang:
    data = {
        'hin': {
            'vowels': dict(अ=None, आ='ा', इ='ि', ई='ी', उ='ु', ऊ='ू', ऋ='ृ',
                           ॠ='ॄ', ऍ='ॅ', ऎ='ॆ', ए='े', ऐ='ै', ऑ='ॉ', ऒ='ॊ',
                           ओ='ो', औ='ौ', ऌ='ॢ', ॡ='ॣ'),
            'consonants': set(range(ord('क'), ord('ह') + 1)) | set(range(ord('क़'), ord('य़') + 1)),
            'semi_vowels': {'ं', 'ः', 'ँ', '़'},  # figure out how to use these in version 2
            'impl_vowel': 'अ',
            'virama': '्'
        }
    }
    data['sanskrit'] = data['devanagari'] = data['hindi'] = data['hin']
    assert lang_name in data, f'{lang_name} is not supported'
    lang = data[lang_name]
    return AbugidaLang(lang_name, **lang)


def main(inp, out, lang, decode=False):
    lang = get_aubugida(lang) if isinstance(lang, str) else lang
    lines = (lang.decode_all if decode else lang.encode_all)(inp)
    for line in lines:
        out.write(line)
        out.write('\n')

if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', '--inp', type=argparse.FileType('r', encoding='utf8', errors='ignore'),
                   default=sys.stdin, help='Input file path')
    p.add_argument('-o', '--out', type=argparse.FileType('w', encoding='utf8', errors='ignore'),
                   default=sys.stdout,
                   help='Output file path')
    p.add_argument('-d', '--decode', action='store_true',
                   help="Restore the standard unicode stream")
    p.add_argument('-l', '--lang', choices=['hin', 'devanagari'], help="Language identifier", required=True)
    args = vars(p.parse_args())
    main(**args)
