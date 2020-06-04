from flask import Flask, render_template_string, request
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch import set_grad_enabled
from rtg import TranslationExperiment as Experiment, log
from rtg.module.decoder import Decoder

# TODO: eventually needs to be AJAX?


def main():
    cli_args = parse_args()
    decoder, dec_args = prepare_decoder(cli_args)
    app = Flask(__name__)
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
    # relevant for server?:
    # parser.add_argument( "-b", "--batch-size", atype=int, help="batch size for 1 beam. effective_batch = batch_size/beam_size", )
    parser.add_argument(
        "-msl",
        "--max-src-len",
        type=int,
        help="max source len; longer seqs will be truncated",
    )
    args = vars(parser.parse_args())
    return args


def attach_translate_route(app, decoder, dec_args):
    @app.route("/translate", methods=["GET", "POST"])
    def translate():
        form = '<form method="POST"><input type="text" name="translateme"><button type="submit">Submit</button></form>'
        if request.method == "POST":
            line = request.form["translateme"]
            translated = decoder.decode_sentence(line, **dec_args)[0][1]
            log.info("Decode :: {line} -> {translated}")
            return render_template_string(form + f"<p>{translated}</p>")
        return render_template_string(form)


def validate_args(cli_args, conf_args, exp: Experiment):
    if not cli_args.pop("skip_check"):  # if --skip-check is not requested
        assert (
            exp.has_prepared()
        ), f'Experiment dir {exp.work_dir} is not ready to train. Please run "prep" sub task'
        assert exp.has_trained(), (
            f"Experiment dir {exp.work_dir} is not ready to decode."
            f' Please run "train" sub task or --skip-check to ignore this'
        )
    # useful?:
    # if cli_args.get("batch_size"):
    #     batch_size = cli_args["batch_size"] / conf_args.get("beam_size", 1)
    #     log.info(f"Batch size is {batch_size}")
    #     conf_args["batch_size"] = batch_size
    if cli_args.get("max_src_len"):
        conf_args["max_src_len"] = cli_args["max_src_len"]


if __name__ == "__main__":
    main()
