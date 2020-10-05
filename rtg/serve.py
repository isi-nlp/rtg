#!/usr/bin/env python
"""
Serves an RTG model using Flask HTTP server
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from flask import Flask, request, jsonify
import torch

from rtg import TranslationExperiment as Experiment, log
from rtg.module.decoder import Decoder


def prepare_decoder(cli_args):
    # No grads required for decode
    torch.set_grad_enabled(False)
    exp = Experiment(cli_args.pop("exp_dir"), read_only=True)
    dec_args = exp.config.get("decoder") or exp.config["tester"].get("decoder", {})
    validate_args(cli_args, dec_args, exp)
    decoder = Decoder.new(exp, ensemble=dec_args.pop("ensemble", 1))
    return decoder, dec_args


def attach_translate_route(app, decoder, dec_args):

    app.config['JSON_AS_ASCII'] = False

    @app.route("/translate", methods=["POST", "GET"])
    def translate():
        if request.method not in ("POST", "GET"):
            return "GET and POST are supported", 400
        if request.method == 'GET':
            sources = request.args.getlist("source", None)
        else:
            sources = request.form.getlist("source", None)
        if not sources:
            return "Please provide parameter 'source'", 400

        translations = []
        for source in sources:
            translated = decoder.decode_sentence(source, **dec_args)[0][1]
            translations.append(translated)
        res = dict(source=sources, translation=translations)
        return jsonify(res)


def validate_args(cli_args, conf_args, exp: Experiment):
    if not cli_args.pop("skip_check"):  # if --skip-check is not requested
        assert exp.has_prepared(), (f'Experiment dir {exp.work_dir} is not ready to train.'
                                    f' Please run "prep" sub task')
        assert exp.has_trained(), (f"Experiment dir {exp.work_dir} is not ready to decode."
                                   f" Please run 'train' sub task or --skip-check to ignore this")

def parse_args():
    parser = ArgumentParser(
        prog="rtg.serve",
        description="deploy a model to a RESTful react server",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("exp_dir", help="Experiment directory", type=str)
    parser.add_argument("-sc", "--skip-check", action="store_true",
        help="Skip Checking whether the experiment dir is prepared and trained")
    parser.add_argument("--debug", action="store_true", help="Run Flask server in debug mode")
    parser.add_argument("-p", "--port", type=int, help="port to run server on", default=6060)
    parser.add_argument("-ho", "--host", help="Host address to bind.", default='0.0.0.0')
    parser.add_argument("-msl", "--max-src-len", type=int,
                        help="max source len; longer seqs will be truncated")
    args = vars(parser.parse_args())
    return args


def main():
    cli_args = parse_args()
    decoder, dec_args = prepare_decoder(cli_args)
    app = Flask(__name__)
    #CORS(app)  # TODO: insecure
    if cli_args.pop('debug'):
        app.debug = True
    attach_translate_route(app, decoder, dec_args)
    app.run(port=cli_args["port"], host=cli_args["host"])


if __name__ == "__main__":
    main()
