#!/usr/bin/env python
"""
Serves an RTG model using Flask HTTP server
"""
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from flask import Flask, request, jsonify, render_template, send_from_directory, Blueprint

from rtg import TranslationExperiment as Experiment
from rtg.module.decoder import Decoder

torch.set_grad_enabled(False)

exp = None
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

bp = Blueprint('burritos', __name__, template_folder='templates')


@bp.route('/')
def index():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(bp.root_path, 'static', 'favicon'), 'favicon.ico')


def attach_translate_route(cli_args):
    global exp, src_prep, tgt_postp
    exp = Experiment(cli_args.pop("exp_dir"), read_only=True)
    dec_args = exp.config.get("decoder") or exp.config["tester"].get("decoder", {})
    decoder = Decoder.new(exp, ensemble=dec_args.pop("ensemble", 1))
    src_prep = exp.get_pre_transform(side='src')
    tgt_postp = exp.get_post_transform(side='tgt')

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
                translated = tgt_postp(translated)
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
    # CORS(app)  # TODO: insecure
    app.run(port=cli_args["port"], host=cli_args["host"])
    # A very useful tutorial is found at:
    # https://www.digitalocean.com/community/tutorials/how-to-make-a-web-application-using-flask-in-python-3


if __name__ == "__main__":
    main()
