import argparse
from flask import Flask
from torch import load
from rtg.module.decoder import instantiate_model

app = Flask(__name__)
​
@app.route("/")
def hello():
    return "Image classification example\n"

def parse_args():
    parser = argparse.ArgumentParser(description="deploy model to RESTful react server")
    parser.add_argument(
        "model", help="Picked model file", type=Path
    )  # TODO: switch to conf file
    args = parser.parse_args()
    return args.model

if __name__ == "__main__":
    path = parse_args()
    checkpt_state = load(path)
    model = instantiate_model(checkpt_state)
