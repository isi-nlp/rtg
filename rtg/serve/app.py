#!/usr/bin/env python
"""
Serves an RTG model using Flask HTTP server
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from flask import Flask, request, jsonify, render_template, send_from_directory, Blueprint
import torch
import os
import html
from sacremoses import MosesTokenizer, MosesDetokenizer, MosesPunctNormalizer, MosesTruecaser
from functools import partial

from rtg import TranslationExperiment as Experiment
from rtg.module.decoder import Decoder
from rtg.utils import shell_pipe


torch.set_grad_enabled(False)

transformers  = {
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
        #self.pipeline = [transformers[key] for key in chain]

    def __call__(self, text):
        res = text
        for stage in self.chain:
            res = stage(res)
        return res

    @classmethod
    def make(cls, names):
        chain = []
        for name in names:
            if name.startswith("#!"): # shell
                cmd_line = name[2:].strip()
                chain.append(lambda x: shell_pipe(cmd_line=cmd_line, input=x))
            elif name in transformers:
                chain.append(transformers[name])
            else:
                raise Exception(f'Text transformer "{name}" unknown; Known: {transformers.keys()}'
                                f'\n Also, you may use shell commandline prefixing the hasbang "#!"'
                                f'\nExample: #!tokenizer.perl'
                                f'\n    #!/path/to/tokenizer.perl -en | sed \'/<old>/<new>/\'')
        return cls(chain=chain)

    # preprocessor used for 500 Eng
    @classmethod
    def recommended(cls) -> ('TextTransform', 'TextTransform'):
        pre_proc = cls.make(names=['html_unescape', 'punct_norm', 'moses_tok'])
        post_proc = cls.make(names=['moses_detok', 'drop_unk'])
        return pre_proc, post_proc

    @classmethod
    def basic(cls) -> ('TextTransform', 'TextTransform'):
        pre_proc = cls.make(names=['space_tok'])
        post_proc = cls.make(names=['space_detok', 'drop_unk'])
        return pre_proc, post_proc


class RtgIO:

    def __init__(self, exp):
        self.exp = exp
        self.tokr = MosesTokenizer()
        self.detokr = MosesDetokenizer()
        self.punct_normr = MosesPunctNormalizer()
        #self.true_caser = MosesTruecaser()

        self.punct_normalize = True
        self.tokenize = True
        self.html_unesc = True
        self.drop_unks = True
        #self.truecase = True
        self.detokenize = True

    def pre_process(self, text):
        # Any pre-processing on input
        if self.html_unesc:
            text = html.unescape(text)
        if self.punct_normalize:
            text = self.punct_normr.normalize(text)
        if self.tokenize:
            text = self.tokr.tokenize(text, escape=False, return_str=True,
                                      aggressive_dash_splits=True)
            # protected_patterns=self.tokr.WEB_PROTECTED_PATTERNS
        return text

    def post_process(self, tokens):
        # Any post-processing on output
        assert isinstance(tokens, list)
        if self.detokenize:
            text = self.detokr.detokenize(tokens=tokens, return_str=True, unescape=True)
        else:
            text = " ".join(tokens)
        if self.drop_unks:
            text = text.replace("<unk>", "")
        #if self.truecase:
        #    text = self.true_caser.truecase(text, return_str=True)
        return text


exp = None
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

bp = Blueprint('burritos', __name__, template_folder='templates')

@bp.route('/')
def index():
    #return "this is a test"
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(bp.root_path, 'static', 'favicon'), 'favicon.ico')

def attach_translate_route(cli_args):
    global exp, src_prep, tgt_postp
    exp = Experiment(cli_args.pop("exp_dir"), read_only=True)
    dec_args = exp.config.get("decoder") or exp.config["tester"].get("decoder", {})
    decoder = Decoder.new(exp, ensemble=dec_args.pop("ensemble", 1))
    src_prep, tgt_postp = TextTransform.recommended()
    src_prep_chain = exp.config.get('prep', {}).get('src_pre_proc', None)
    tgt_postp_chain = exp.config.get('prep', {}).get('tgt_post_proc', None)
    if src_prep_chain:
        src_prep = TextTransform.make(names=src_prep_chain)
    if tgt_postp_chain:
        tgt_postp = TextTransform.make(names=tgt_postp_chain)

    @bp.route("/translate", methods=["POST", "GET"])
    def translate():
        if request.method not in ("POST", "GET"):
            return "GET and POST are supported", 400
        if request.method == 'GET':
            sources = request.args.getlist("source", None)
        else:
            sources = (request.json or {}).get('source', None) or request.form.getlist("source")
            if isinstance(sources, str):
                sources = [sources]
        if not sources:
            return "Please submit 'source' parameter", 400
        prep = request.args.get('prep', "True").lower() in ("true", "yes", "y", "t")
        if prep:
            sources = [src_prep(sent) for sent in sources]
        translations = []
        for source in sources:
            translated = decoder.decode_sentence(source, **dec_args)[0][1]
            if prep:
                translated = tgt_postp(translated.split())
            translations.append(translated)

        res = dict(source=sources, translation=translations)
        return jsonify(res)

    @bp.route("/conf.yml", methods=["GET"])
    def get_conf():
        conf_str = exp._config_file.read_text(encoding='utf-8', errors='ignore')
        return render_template('conf.yml.html', conf_str=conf_str)

    @bp.route("/about", methods=["GET"])
    def about():
        def_desc = "Model description is unavailable; please update conf.yml"
        return render_template('about.html', model_desc=exp.config.get("description", def_desc))


def parse_args():
    parser = ArgumentParser(
        prog="rtg.serve",
        description="Deploy an RTG model to a RESTful server",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("exp_dir", help="Experiment directory", type=str)
    parser.add_argument("-d", "--debug", action="store_true", help="Run Flask server in debug mode")
    parser.add_argument("-p", "--port", type=int, help="port to run server on", default=6060)
    parser.add_argument("-ho", "--host", help="Host address to bind.", default='0.0.0.0')
    parser.add_argument("-b", "--base", help="Base prefix path for all the URLs")
    parser.add_argument("-msl", "--max-src-len", type=int, default=250,
                        help="max source len; longer seqs will be truncated")
    args = vars(parser.parse_args())
    return args

# uwsgi can take CLI args too
# uwsgi --http 127.0.0.1:5000 --module rtg.serve.app:app --pyargv "rtgv0.5-768d9L6L-512K64K-datav1"
cli_args = parse_args()
attach_translate_route(cli_args)
app.register_blueprint(bp, url_prefix=cli_args.get('base'))
if cli_args.pop('debug'):
    app.debug = True


# register a home page if needed
if cli_args.get('base'):
    @app.route('/')
    def home():
        return render_template('home.html', demo_url=cli_args.get('base'))

def main():
    #CORS(app)  # TODO: insecure
    app.run(port=cli_args["port"], host=cli_args["host"])

    # A very useful tutorial is found at:
    # https://www.digitalocean.com/community/tutorials/how-to-make-a-web-application-using-flask-in-python-3

if __name__ == "__main__":
    main()
