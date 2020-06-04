from flask import (
    Flask,
    render_template_string,
    request,
    jsonify,
    Response,
    make_response,
)
from flask_cors import CORS

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch import set_grad_enabled
from rtg import TranslationExperiment as Experiment, log
from rtg.module.decoder import Decoder


def main():
    cli_args = parse_args()
    decoder, dec_args = prepare_decoder(cli_args)
    app = Flask(__name__)
    CORS(app)  # TODO: insecure
    app.debug = True
    attach_translate_route(app, decoder, dec_args)
    app.run(port=cli_args.get("port"))


def prepare_decoder(cli_args):
    # No grads required for decode
    set_grad_enabled(False)
    exp = Experiment(cli_args.pop("exp_dir"), read_only=True)
    dec_args = exp.config.get("decoder") or exp.config["tester"].get("decoder", {})
    validate_args(cli_args, dec_args, exp)
    decoder = Decoder.new(exp, ensemble=dec_args.pop("ensemble", 1))
    return decoder, dec_args


def parse_args():
    parser = ArgumentParser(
        prog="rtg.deploy",
        description="deploy a model to a RESTful react server",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("exp_dir", help="Experiment directory", type=str)
    parser.add_argument(
        "-sc",
        "--skip-check",
        action="store_true",
        help="Skip Checking whether the experiment dir is prepared and trained",
    )
    parser.add_argument(
        "-p", "--port", type=int, help="port to run server on", default=5000
    )
    parser.add_argument(
        "-msl",
        "--max-src-len",
        type=int,
        help="max source len; longer seqs will be truncated",
    )
    args = vars(parser.parse_args())
    return args


def attach_translate_route(app, decoder, dec_args):
    @app.route("/translate", methods=["POST"])
    def translate():
        if request.method != "POST":
            return
        json = request.get_json(force=True)
        line = json.get("translateme")
        translated = decoder.decode_sentence(line, **dec_args)[0][1]
        # log.info(f"Decode :: {line} -> {translated}")
        return jsonify({"result": translated})

    # Example request:
    ## var f = await fetch("http://localhost:5000/translate", {
    ##     method: 'POST',
    ##     body: JSON.stringify({"translateme": "C’est simple comme bonjour"})})
    ## var json = await f.json()
    ## console.log(json.result)


def validate_args(cli_args, conf_args, exp: Experiment):
    if not cli_args.pop("skip_check"):  # if --skip-check is not requested
        assert (
            exp.has_prepared()
        ), f'Experiment dir {exp.work_dir} is not ready to train. Please run "prep" sub task'
        assert exp.has_trained(), (
            f"Experiment dir {exp.work_dir} is not ready to decode."
            f' Please run "train" sub task or --skip-check to ignore this'
        )


if __name__ == "__main__":
    main()
