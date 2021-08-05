#!/usr/bin/env bash

awkg -F '\t' -b '
from html import unescape
from sacremoses import MosesTokenizer
from sacremoses import MosesPunctNormalizer

from functools import partial
tokr = partial(MosesTokenizer(lang="en").tokenize, return_str=False,
     aggressive_dash_splits=True, escape=False)
#  protected_patterns=MosesTokenizer.BASIC_PROTECTED_PATTERNS)


normr = MosesPunctNormalizer().normalize

goods, bads = 0, 0
src_len=[3, 120]; tgt_len=[3, 120]
' '
good = len(R) == 2
if good:
  src = R[0].strip()
  tgt = R[1].strip()
  good &= "http" not in src and "http" not in tgt
  # unescape html/xml -> normalize puncts -> tokenize
  src = tokr(normr(unescape(src)))
  tgt = tokr(normr(unescape(tgt)))
  good &= src_len[0] <= len(src) <= src_len[1] and tgt_len[0] <= len(tgt) <= tgt_len[1]
  good &= 1/5 <= len(src)/(0.001+len(tgt)) <= 5
  good &= max(len(w) for w in src + tgt) < 30 

if good:
   goods += 1
   print(" ".join(src), " ".join(tgt))
else:
   bads += 1
   #sys.stderr.write(" ".join(src), " ".join(tgt))

# if not good: # print bad
#  print(R0)
' -e 'sys.stderr.write(f"good={goods:,} bad={bads:,} records")'
