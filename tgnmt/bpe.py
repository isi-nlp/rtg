# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code is taken from Tensor2Tensor
# and performed a few simplifications

import collections
from itertools import chain
import re
import os
from typing import List, Tuple
from tgnmt import log
from tgnmt.dataprep import RESERVED_TOKS

# Reserved tokens for things like padding and EOS symbols.
RESERVED_TOKENS = [tok for tok, idx in RESERVED_TOKS]
NUM_RESERVED_TOKENS = len(RESERVED_TOKS)

# Regular expression for unescaping token strings.
# '\u' is converted to '_'
# '\\' is converted to '\'
# '\213;' is converted to unichr(213)
_UNESCAPE_REGEX = re.compile(r"\\u|\\\\|\\([0-9]+);")
_ESCAPE_CHARS = set(u"\\_u;0123456789")


class Tokenizer:
    """A naive white space tokenizer"""

    @staticmethod
    def encode(text):
        """Encode a unicode string as a list of tokens.

        Args:
          text: a unicode string
        Returns:
          a list of tokens as Unicode strings
        """
        return text.split()

    @staticmethod
    def decode(tokens):
        """Decode a list of tokens to a unicode string.

        Args:
          tokens: a list of Unicode strings
        Returns:
          a unicode string
        """
        return " ".join(tokens)


def _escape_token(token, alphabet):
    """Escape away underscores and OOV characters and append '_'.
    This allows the token to be expressed as the concatenation of a list
    of subtokens from the vocabulary. The underscore acts as a sentinel
    which allows us to invertibly concatenate multiple such lists.
    Args:
      token: A unicode string to be escaped.
      alphabet: A set of all characters in the vocabulary's alphabet.
    Returns:
      escaped_token: An escaped unicode string.
    Raises:
      ValueError: If the provided token is not unicode.
    """
    if type(token) is not str:
        raise ValueError("Expected string type for token, got %s" % type(token))

    token = token.replace(u"\\", u"\\\\").replace(u"_", u"\\u")
    ret = [c if c in alphabet and c != u"\n" else r"\%d;" % ord(c) for c in token]
    return u"".join(ret) + "_"


def _unescape_token(escaped_token):
    """Inverse of _escape_token().
    Args:
      escaped_token: a unicode string
    Returns:
      token: a unicode string
    """

    def match(m):
        if m.group(1) is None:
            return u"_" if m.group(0) == u"\\u" else u"\\"

        try:
            return chr(int(m.group(1)))
        except (ValueError, OverflowError) as _:
            return u"\u3013"  # Unicode for undefined character.

    trimmed = escaped_token[:-1] if escaped_token.endswith("_") else escaped_token
    return _UNESCAPE_REGEX.sub(match, trimmed)

# TG: TODO: simplify this class, not all this functionality needed for non-Googlers


class SubwordTextEncoder:
    """Class for invertibly encoding text using a limited vocabulary.
    Invertibly encodes a native string as a sequence of subtokens from a limited
    vocabulary.
    A SubwordTextEncoder is built from a corpus (so it is tailored to the text in
    the corpus), and stored to a file. See text_encoder_build_subword.py.
    It can then be loaded and used to encode/decode any text.
    Encoding has four phases:
    1. Tokenize into a list of tokens.  Each token is a unicode string of either
       all alphanumeric characters or all non-alphanumeric characters.  We drop
       tokens consisting of a single space that are between two alphanumeric
       tokens.
    2. Escape each token.  This escapes away special and out-of-vocabulary
       characters, and makes sure that each token ends with an underscore, and
       has no other underscores.
    3. Represent each escaped token as a the concatenation of a list of subtokens
       from the limited vocabulary.  Subtoken selection is done greedily from
       beginning to end.  That is, we construct the list in order, always picking
       the longest subtoken in our vocabulary that matches a prefix of the
       remaining portion of the encoded token.
    4. Concatenate these lists.  This concatenation is invertible due to the
       fact that the trailing underscores indicate when one list is finished.
    """

    def __init__(self, filename=None, subtoks=None):
        """Initialize and read from a file, if provided.
        Args:
          filename: filename from which to read vocab. If None, do not load a
            vocab
        """
        self._alphabet = set()
        self.filename = filename
        if filename is not None:
            self._load_from_file(filename)
        if subtoks is not None:
            self._init_from_subtoks(subtoks)
        super(SubwordTextEncoder, self).__init__()

    def encode(self, s):
        """Converts a native string to a list of subtoken ids.
        Args:
          s: a native string.
        Returns:
          a list of integers in the range [0, vocab_size)
        """
        return self._tokens_to_subtoken_ids(Tokenizer.encode(s))

    def encode_without_tokenizing(self, token_text):
        """Converts string to list of subtoken ids without calling tokenizer.
        This treats `token_text` as a single token and directly converts it
        to subtoken ids. This may be useful when the default tokenizer doesn't
        do what we want (e.g., when encoding text with tokens composed of lots of
        nonalphanumeric characters). It is then up to the caller to make sure that
        raw text is consistently converted into tokens. Only use this if you are
        sure that `encode` doesn't suit your needs.
        Args:
          token_text: A native string representation of a single token.
        Returns:
          A list of subword token ids; i.e., integers in the range [0, vocab_size).
        """
        return self._tokens_to_subtoken_ids([token_text])

    def decode(self, ids):
        """Converts a sequence of subtoken ids to a native string.
        Args:
          ids: a list of integers in the range [0, vocab_size)
        Returns:
          a native string
        """
        return Tokenizer.decode(self._subtoken_ids_to_tokens(ids))

    def decode_list(self, ids):
        return [self._subtoken_id_to_subtoken_string(s) for s in ids]

    @property
    def vocab_size(self):
        """The subtoken vocabulary size."""
        return len(self._all_subtoken_strings)

    def _tokens_to_subtoken_ids(self, tokens):
        """Converts a list of tokens to a list of subtoken ids.
        Args:
          tokens: a list of strings.
        Returns:
          a list of integers in the range [0, vocab_size)
        """
        ret = []
        for token in tokens:
            ret.extend(self._token_to_subtoken_ids(token))
        return ret

    def tokens_to_subtokens(self, tokens: List[str]) -> List[str]:
        """Converts a list of tokens to a list of subtokens.
        Args:
          tokens: a list of strings.
        Returns:
          a list of integers in the range [0, vocab_size)
        """
        ret = []
        for token in tokens:
            ret.extend(self.token_to_subtokens(token))
        return ret

    def _token_to_subtoken_ids(self, token):
        """Converts token to a list of subtoken ids.
        Args:
          token: a string.
        Returns:
          a list of integers in the range [0, vocab_size)
        """
        cache_location = hash(token) % self._cache_size
        cache_key, cache_value = self._cache[cache_location]
        if cache_key == token:
            return cache_value
        ret = self._escaped_token_to_subtoken_ids(_escape_token(token, self._alphabet))
        self._cache[cache_location] = (token, ret)
        return ret

    def token_to_subtokens(self, token: str) -> List[str]:
        """Converts token to a list of subtokens.
        Args:
          token: a string.
        Returns:
          a list of sub tokens
        """
        cache_location = hash(token) % self._cache_size
        cache_key, cache_value = self._cache_str[cache_location]
        if cache_key == token:
            return cache_value
        escaped_token = _escape_token(token, self._alphabet)
        ret = self._escaped_token_to_subtoken_strings(escaped_token)
        self._cache_str[cache_location] = (token, ret)
        return ret

    def _subtoken_ids_to_tokens(self, subtokens):
        """Converts a list of subtoken ids to a list of tokens.
        Args:
          subtokens: a list of integers in the range [0, vocab_size)
        Returns:
          a list of strings.
        """
        return self.un_split([self._subtoken_id_to_subtoken_string(s) for s in subtokens])

    @staticmethod
    def un_split(subtokens: List[str]):
        """Reverse sub word split operation, i.e combine subwords into full words"""
        splits = ''.join(subtokens).split("_")
        splits = [_unescape_token(tok + "_") for tok in splits if tok]
        splits = [split for split in splits if split]
        return splits

    def _subtoken_id_to_subtoken_string(self, subtoken):
        """Converts a subtoken integer ID to a subtoken string."""
        if 0 <= subtoken < self.vocab_size:
            return self._all_subtoken_strings[subtoken]
        return u""

    def _escaped_token_to_subtoken_strings(self, escaped_token: str) -> List[str]:
        """Converts an escaped token string to a list of subtoken strings.
        Args:
          escaped_token: An escaped token as a unicode string.
        Returns:
          A list of subtokens as unicode strings.
        """
        # NOTE: This algorithm is greedy; it won't necessarily produce the "best"
        # list of subtokens.
        ret = []
        start = 0
        token_len = len(escaped_token)
        while start < token_len:
            for end in range(min(token_len, start + self._max_subtoken_len), start, -1):
                subtoken = escaped_token[start:end]
                if subtoken in self._subtoken_string_to_id:
                    ret.append(subtoken)
                    start = end
                    break

            else:  # Did not break
                # If there is no possible encoding of the escaped token then one of the
                # characters in the token is not in the alphabet. This should be
                # impossible and would be indicative of a bug.
                assert False, "Token substring not found in subtoken vocabulary."

        return ret

    def _escaped_token_to_subtoken_ids(self, escaped_token):
        """Converts an escaped token string to a list of subtoken IDs.
        Args:
          escaped_token: An escaped token as a unicode string.
        Returns:
          A list of subtoken IDs as integers.
        """
        return [
            self._subtoken_string_to_id[subtoken]
            for subtoken in self._escaped_token_to_subtoken_strings(escaped_token)
        ]

    @classmethod
    def build_from_generator(cls,
                             generator,
                             target_vocab_size,
                             max_subtoken_length=None,
                             reserved_tokens=None):
        """Builds a SubwordTextEncoder from the generated text.
        Args:
          generator: yields text.
          target_vocab_size: int, approximate vocabulary size to create.
          max_subtoken_length: Maximum length of a subtoken. If this is not set,
            then the runtime and memory use of creating the vocab is quadratic in
            the length of the longest token. If this is set, then it is instead
            O(max_subtoken_length * length of longest token).
          reserved_tokens: List of reserved tokens. The global variable
            `RESERVED_TOKENS` must be a prefix of `reserved_tokens`. If this
            argument is `None`, it will use `RESERVED_TOKENS`.
        Returns:
          SubwordTextEncoder with `vocab_size` approximately `target_vocab_size`.
        """
        token_counts = collections.defaultdict(int)
        for item in generator:
            for tok in Tokenizer.encode(item):
                token_counts[tok] += 1
        encoder = cls.build_to_target_size(target_vocab_size, token_counts, 1, 1e3,
                                           max_subtoken_length=max_subtoken_length,
                                           reserved_tokens=reserved_tokens)
        return encoder

    @classmethod
    def build_to_target_size(cls,
                             target_size,
                             token_counts,
                             min_val,
                             max_val,
                             max_subtoken_length=None,
                             reserved_tokens=None,
                             num_iterations=4):
        """Builds a SubwordTextEncoder that has `vocab_size` near `target_size`.
        Uses simple recursive binary search to find a minimum token count that most
        closely matches the `target_size`.
        Args:
          target_size: Desired vocab_size to approximate.
          token_counts: A dictionary of token counts, mapping string to int.
          min_val: An integer; lower bound for the minimum token count.
          max_val: An integer; upper bound for the minimum token count.
          max_subtoken_length: Maximum length of a subtoken. If this is not set,
            then the runtime and memory use of creating the vocab is quadratic in
            the length of the longest token. If this is set, then it is instead
            O(max_subtoken_length * length of longest token).
          reserved_tokens: List of reserved tokens. The global variable
            `RESERVED_TOKENS` must be a prefix of `reserved_tokens`. If this
            argument is `None`, it will use `RESERVED_TOKENS`.
          num_iterations: An integer; how many iterations of refinement.
        Returns:
          A SubwordTextEncoder instance.
        Raises:
          ValueError: If `min_val` is greater than `max_val`.
        """
        if min_val > max_val:
            raise ValueError("Lower bound for the minimum token count is greater than the upper bound.")
        if target_size < 1:
            raise ValueError("Target size must be positive.")

        if reserved_tokens is None:
            reserved_tokens = RESERVED_TOKENS

        def bisect(min_val, max_val):
            """Bisection to find the right size."""
            present_count = (max_val + min_val) // 2
            log.info("Trying min_count %d" % present_count)
            subtokenizer = cls()
            subtokenizer.build_from_token_counts(
                token_counts, present_count, num_iterations,
                max_subtoken_length=max_subtoken_length,
                reserved_tokens=reserved_tokens)

            # Being within 1% of the target size is ok.
            is_ok = abs(subtokenizer.vocab_size - target_size) * 100 < target_size
            # If min_val == max_val, we can't do any better than this.
            if is_ok or min_val >= max_val or present_count < 2:
                return subtokenizer

            if subtokenizer.vocab_size > target_size:
                other_subtokenizer = bisect(present_count + 1, max_val)
            else:
                other_subtokenizer = bisect(min_val, present_count - 1)

            if other_subtokenizer is None:
                return subtokenizer

            if abs(other_subtokenizer.vocab_size - target_size) < abs(subtokenizer.vocab_size - target_size):
                return other_subtokenizer
            return subtokenizer

        return bisect(min_val, max_val)

    def build_from_token_counts(self,
                                token_counts,
                                min_count,
                                num_iterations=4,
                                reserved_tokens=None,
                                max_subtoken_length=None):
        """Train a SubwordTextEncoder based on a dictionary of word counts.
        Args:
          token_counts: a dictionary of Unicode strings to int.
          min_count: an integer - discard subtokens with lower counts.
          num_iterations: an integer.  how many iterations of refinement.
          reserved_tokens: List of reserved tokens. The global variable
            `RESERVED_TOKENS` must be a prefix of `reserved_tokens`. If this
            argument is `None`, it will use `RESERVED_TOKENS`.
          max_subtoken_length: Maximum length of a subtoken. If this is not set,
            then the runtime and memory use of creating the vocab is quadratic in
            the length of the longest token. If this is set, then it is instead
            O(max_subtoken_length * length of longest token).
        Raises:
          ValueError: if reserved is not 0 or len(RESERVED_TOKENS). In this case, it
            is not clear what the space is being reserved for, or when it will be
            filled in.
        """
        if reserved_tokens is None:
            reserved_tokens = RESERVED_TOKENS
        else:
            # There is not complete freedom in replacing RESERVED_TOKENS.
            for default, proposed in zip(RESERVED_TOKENS, reserved_tokens):
                if default != proposed:
                    raise ValueError("RESERVED_TOKENS must be a prefix of "
                                     "reserved_tokens.")

        # Initialize the alphabet. Note, this must include reserved tokens or it can
        # result in encoding failures.
        alphabet_tokens = chain(token_counts.keys(), reserved_tokens)

        self._init_alphabet_from_tokens(alphabet_tokens)

        # Bootstrap the initial list of subtokens with the characters from the
        # alphabet plus the escaping characters.
        self._init_subtokens_from_list(list(self._alphabet), reserved_tokens=reserved_tokens)

        # We build iteratively.  On each iteration, we segment all the words,
        # then count the resulting potential subtokens, keeping the ones
        # with high enough counts for our new vocabulary.
        if min_count < 1:
            min_count = 1
        for i in range(num_iterations):
            log.info("Iteration {0}".format(i))

            # Collect all substrings of the encoded token that break along current
            # subtoken boundaries.
            subtoken_counts = collections.defaultdict(int)
            for token, count in token_counts.items():
                escaped_token = _escape_token(token, self._alphabet)
                subtokens = self._escaped_token_to_subtoken_strings(escaped_token)
                start = 0
                for subtoken in subtokens:
                    last_position = len(escaped_token) + 1
                    if max_subtoken_length is not None:
                        last_position = min(last_position, start + max_subtoken_length)

                    for end in range(start + 1, last_position):
                        new_subtoken = escaped_token[start:end]
                        subtoken_counts[new_subtoken] += count
                    start += len(subtoken)

            # Array of sets of candidate subtoken strings, by length.
            len_to_subtoken_strings = []
            for subtoken_string, count in subtoken_counts.items():
                lsub = len(subtoken_string)
                if count >= min_count:
                    while len(len_to_subtoken_strings) <= lsub:
                        len_to_subtoken_strings.append(set())
                    len_to_subtoken_strings[lsub].add(subtoken_string)

            # Consider the candidates longest to shortest, so that if we accept
            # a longer subtoken string, we can decrement the counts of its prefixes.
            new_subtoken_strings = []
            for lsub in range(len(len_to_subtoken_strings) - 1, 0, -1):
                subtoken_strings = len_to_subtoken_strings[lsub]
                for subtoken_string in subtoken_strings:
                    count = subtoken_counts[subtoken_string]
                    if count >= min_count:
                        # Exclude alphabet tokens here, as they must be included later,
                        # explicitly, regardless of count.
                        if subtoken_string not in self._alphabet:
                            new_subtoken_strings.append((count, subtoken_string))
                        for l in range(1, lsub):
                            subtoken_counts[subtoken_string[:l]] -= count

            # Include the alphabet explicitly to guarantee all strings are encodable.
            new_subtoken_strings.extend((subtoken_counts.get(a, 0), a)
                                        for a in self._alphabet)
            new_subtoken_strings.sort(reverse=True)

            # Reinitialize to the candidate vocabulary.
            new_subtoken_strings = [subtoken for _, subtoken in new_subtoken_strings]
            if reserved_tokens:
                new_subtoken_strings = reserved_tokens + new_subtoken_strings

            self._init_subtokens_from_list(new_subtoken_strings)
            log.info("vocab_size = %d" % self.vocab_size)

    @property
    def all_subtoken_strings(self):
        return tuple(self._all_subtoken_strings)

    def dump(self):
        """Debugging dump of the current subtoken vocabulary."""
        subtoken_strings = [(i, s) for s, i in self._subtoken_string_to_id.items()]
        print(u", ".join(f"{i} : '{s}'" for i, s in sorted(subtoken_strings)))

    def _init_subtokens_from_list(self, subtoken_strings, reserved_tokens=None):
        """Initialize token information from a list of subtoken strings.
        Args:
          subtoken_strings: a list of subtokens
          reserved_tokens: List of reserved tokens. We must have `reserved_tokens`
            as None or the empty list, or else the global variable `RESERVED_TOKENS`
            must be a prefix of `reserved_tokens`.
        Raises:
          ValueError: if reserved is not 0 or len(RESERVED_TOKENS). In this case, it
            is not clear what the space is being reserved for, or when it will be
            filled in.
        """
        if reserved_tokens is None:
            reserved_tokens = []

        if reserved_tokens:
            self._all_subtoken_strings = reserved_tokens + subtoken_strings
        else:
            self._all_subtoken_strings = subtoken_strings

        # we remember the maximum length of any subtoken to avoid having to
        # check arbitrarily long strings.
        self._max_subtoken_len = max(len(s) for s in subtoken_strings)
        self._subtoken_string_to_id = {
            s: i + len(reserved_tokens) for i, s in enumerate(subtoken_strings) if s
        }

        # Initialize the cache to empty.
        self._cache_size = 2 ** 20
        self._cache = [(None, None)] * self._cache_size

        # Initialize the cache to empty.
        self._cache_str: List[Tuple[str, List[str]]] = [(None, None)] * self._cache_size   # caches subtoken strings

    def _init_alphabet_from_tokens(self, tokens):
        """Initialize alphabet from an iterable of token or subtoken strings."""
        # Include all characters from all tokens in the alphabet to guarantee that
        # any token can be encoded. Additionally, include all escaping characters.
        self._alphabet = {c for token in tokens for c in token}
        self._alphabet |= _ESCAPE_CHARS

    def _load_from_file_object(self, f):
        """Load from a file object.
        Args:
          f: File object to load vocabulary from
        """
        subtoken_strings = []
        for line in f:
            s = line.strip()
            # Some vocab files wrap words in single quotes, but others don't
            if ((s.startswith("'") and s.endswith("'")) or
                    (s.startswith("\"") and s.endswith("\""))):
                s = s[1:-1]
            subtoken_strings.append(s)
        self._init_subtokens_from_list(subtoken_strings)
        self._init_alphabet_from_tokens(subtoken_strings)

    def _init_from_subtoks(self, subtoks: List[str]):
        """
        Initializes an encoder from subtoken strings
        :param subtoks:
        :return:
        """
        self._init_subtokens_from_list(subtoks)
        self._init_alphabet_from_tokens(subtoks)

    def _load_from_file(self, filename):
        """Load from a vocab file."""
        if not os.path.exists(filename):
            raise ValueError("File %s not found" % filename)
        with open(filename) as f:
            self._load_from_file_object(f)

    def store_to_file(self, filename, add_single_quotes=True):
        with open(filename, "w", encoding='utf-8') as f:
            for subtoken_string in self._all_subtoken_strings:
                if add_single_quotes:
                    f.write(f"'{subtoken_string}'\n")
                else:
                    f.write(f"{subtoken_string}\n")