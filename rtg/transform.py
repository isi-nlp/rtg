"""
transformation of text
"""
import html
from sacremoses import MosesTokenizer, MosesDetokenizer, MosesPunctNormalizer, MosesTruecaser
from functools import partial
from rtg.utils import shell_pipe

text_transformers = {
    'no_op': lambda x: x,
    'space_tok': lambda x: ' '.join(x.strip().split()),  # removes extra white spaces
    'space_detok': lambda toks: ' '.join(toks),
    'moses_tok': partial(MosesTokenizer().tokenize, escape=False, return_str=True,
                         aggressive_dash_splits=True,
                         protected_patterns=MosesTokenizer.WEB_PROTECTED_PATTERNS),
    'moses_detok': partial(MosesDetokenizer().detokenize, return_str=True, unescape=True),
    'moses_truecase': partial(MosesTruecaser().truecase, return_str=True),
    'lowercase': lambda x: x.lower(),
    'drop_unk': lambda x: x.replace('<unk>', ''),
    'html_unescape': html.unescape,
    'punct_norm': MosesPunctNormalizer().normalize
}


class TextTransform:

    def __init__(self, chain):
        self.chain = chain
        # self.pipeline = [transformers[key] for key in chain]

    def __call__(self, text, multiple=False):
        res = text
        for stage in self.chain:
            res = stage(res)
        return res

    def map(self, texts):
        yield from (self(text) for text in texts)

    @classmethod
    def make(cls, names):
        chain = []
        for name in names:
            if name.startswith("#!"):  # shell
                cmd_line = name[2:].strip()
                chain.append(lambda x: shell_pipe(cmd_line=cmd_line, input=x))
            elif name in text_transformers:
                chain.append(text_transformers[name])
            else:
                raise Exception(f'Text transformer "{name}" unknown; Known: {text_transformers.keys()}'
                                f'\n Also, you may use shell commandline prefixing the hasbang "#!"'
                                f'\nExample: #!tokenizer.perl'
                                f'\n    #!/path/to/tokenizer.perl -en | sed \'/<old>/<new>/\'')
        return cls(chain=chain)

    @classmethod
    def recommended_pre(cls) -> 'TextTransform':
        # preprocessor used for 500 Eng
        return cls.make(names=['html_unescape', 'punct_norm', 'moses_tok'])

    @classmethod
    def recommended_post(cls) -> 'TextTransform':
        return cls.make(names=['moses_detok', 'drop_unk'])

    @classmethod
    def basic_pre(cls) -> 'TextTransform':
        return cls.make(names=['space_tok'])

    @classmethod
    def basic_post(cls) -> 'TextTransform':
        return cls.make(names=['space_detok', 'drop_unk'])