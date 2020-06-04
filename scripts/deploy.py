from flask import Flask, render_template_string, request

# TODO: eventually needs to be AJAX?

# import argparse
# from torch import load
# from rtg.module.decoder import instantiate_model

app = Flask(__name__)


@app.route("/")
def hello():
    return "Image classification example"


@app.route("/translate", methods=["GET", "POST"])
def translate():
    if request.method == "POST":
        return render_template_string(
            f"<html>{request.form['translateme'][::-1]}</html>"
        )
    return render_template_string(
        '<form method="POST"> <input type="text" name="translateme"> <button type="submit">Submit</button></form>'
    )


# def parse_args():
#     parser = argparse.ArgumentParser(description="deploy model to RESTful react server")
#     parser.add_argument(
#         "model", help="Picked model file", type=Path
#     )  # TODO: switch to conf file
#     args = parser.parse_args()
#     return args.model

# if __name__ == "__main__":
#     path = parse_args()
#     checkpt_state = load(path)
#     model = instantiate_model(checkpt_state)
