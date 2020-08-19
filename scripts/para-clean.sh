#!/usr/bin/env bash

awkg -F '\t' -b 'from html import unescape; goods, bads = 0, 0; src_len=[4, 120]; tgt_len=[4, 120]' '
good = len(R) == 2
if good:
  src = R[0].strip()
  tgt = R[1].strip()
  good &= "http" not in src and "http" not in tgt
  src = unescape(src).split()
  tgt = unescape(tgt).split()
  good &= src_len[0] <= len(src) <= src_len[1] and tgt_len[0] <= len(tgt) <= tgt_len[1]
  good &= 1/5 <= len(src)/(0.001+len(tgt)) <= 5
  good &= max(len(w) for w in src + tgt) < 30 

if good:
   goods += 1
   print(" ".join(src), " ".join(tgt))
else:
   bads += 1

# if not good: # print bad
#  print(R0)
' -e 'sys.stderr.write(f"good={goods:,} bad={bads:,} records")'
