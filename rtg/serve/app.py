#!/usr/bin/env python
"""
Serves an RTG model using Flask HTTP server
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
import torch
import os
from html import unescape
from sacremoses import MosesTokenizer, MosesDetokenizer, MosesPunctNormalizer, MosesTruecaser

from rtg import TranslationExperiment as Experiment
from rtg.module.decoder import Decoder

torch.set_grad_enabled(False)


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
            text = unescape(text)
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


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
exp = None


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static', 'img'), 'favicon.ico')


def attach_translate_route(cli_args):
    global exp
    exp = Experiment(cli_args.pop("exp_dir"), read_only=True)
    dec_args = exp.config.get("decoder") or exp.config["tester"].get("decoder", {})
    decoder = Decoder.new(exp, ensemble=dec_args.pop("ensemble", 1))
    dataprep = RtgIO(exp=exp)

    @app.route("/translate", methods=["POST", "GET"])
    def translate():
        if request.method not in ("POST", "GET"):
            return "GET and POST are supported", 400
        if request.method == 'GET':
            sources = request.args.getlist("source", None)
        else:
            sources = (request.json or {}).get('source', None)
            if isinstance(sources, str):
                sources = [sources]
        if not sources:
            return "Please submit parameter 'source'", 400
        sources = [dataprep.pre_process(sent) for sent in sources]
        translations = []
        for source in sources:
            translated = decoder.decode_sentence(source, **dec_args)[0][1]
            translated = dataprep.post_process(translated.split())
            translations.append(translated)

        res = dict(source=sources, translation=translations)
        return jsonify(res)

    @app.route("/conf.yml", methods=["GET"])
    def get_conf():
        conf_str = exp._config_file.read_text(encoding='utf-8', errors='ignore')
        return render_template('conf.yml.html', conf_str=conf_str)

    @app.route("/about", methods=["GET"])
    def about():
        def_desc = "Model description not available. Please view or update conf.yml"
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
    parser.add_argument("-msl", "--max-src-len", type=int, default=250,
                        help="max source len; longer seqs will be truncated")
    args = vars(parser.parse_args())
    return args

# uwsgi can take CLI args too
# uwsgi --http 127.0.0.1:5000 --module rtg.serve.app:app --pyargv "rtgv0.5-768d9L6L-512K64K-datav1"
cli_args = parse_args()
attach_translate_route(cli_args)


def main():
    #CORS(app)  # TODO: insecure
    if cli_args.pop('debug'):
        app.debug = True

    app.run(port=cli_args["port"], host=cli_args["host"])

    # A very useful tutorial is found at:
    # https://www.digitalocean.com/community/tutorials/how-to-make-a-web-application-using-flask-in-python-3

if __name__ == "__main__":
    main()
