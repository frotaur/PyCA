import os, time, json
from flask import Flask, Response, jsonify, make_response, request
from flask_cors import CORS
import numpy as np
import io, zlib, pickle
import logging

from Automata.models import (
    CA1D, 
    GeneralCA1D, 
    CA2D, 
    Baricelli1D,
    Baricelli2D, 
    LGCA, 
    FallingSand, 
    MultiLenia,
    NCA
)
from Automata.models.ReactionDiffusion import (
    GrayScott, 
    BelousovZhabotinsky, 
    Brusselator
)

# Device to run the automaton
device = 'cuda'

# Define automaton classes without instantiating
automaton_options = {
    "CA2D":         lambda h, w: CA2D((h,w), b_num='3', s_num='23', random=True, device='cuda'),
    "CA1D":         lambda h, w: CA1D((h,w), wolfram_num=90, random=True),
    "GeneralCA1D":  lambda h, w: GeneralCA1D((h,w), wolfram_num=1203, r=3, k=3, random=True),
    "LGCA":         lambda h, w: LGCA((h,w), device='cuda'),
    "Gray-Scott":   lambda h, w: GrayScott((h,w), device='cuda'),
    "Belousov-Zhabotinsky": lambda h, w: BelousovZhabotinsky((h,w), device='cuda'),
    "Brusselator":  lambda h, w: Brusselator((h,w), device='cuda'),
    "Falling Sand": lambda h, w: FallingSand((h,w)),
    "Baricelli 2D": lambda h, w: Baricelli2D((h,w), n_species=7, reprod_collision=True, device='cuda'),
    "Baricelli 1D": lambda h, w: Baricelli1D((h,w), n_species=8, reprod_collision=True),
    "MultiLenia":   lambda h, w: MultiLenia((h,w), param_path='LeniaParams', device='cuda'),
    # "Neural CA":  lambda h, w: NCA((h,w), model_path='NCA_train/trained_model/latestNCA.pt', device='cuda')
}

# Then when initializing the first automaton:
initial_automaton = "CA2D"
stopped = True
# Automaton world size 
W, H = 300, 300
auto = automaton_options[initial_automaton](H, W)


app = Flask(__name__)
app.logger.setLevel(logging.ERROR)
CORS(app, 
    origins=["http://localhost:5173"],
    allow_headers=["Content-Type", "Access-Control-Allow-Headers"],
    expose_headers=["Content-Type"],
    methods=["GET", "POST", "OPTIONS"],
)   

def compress_array(arr):
    s = pickle.dumps(arr, protocol=3)
    e = zlib.compress(s)
    return e

@app.route("/dimensions", methods=["GET"])
def dimensions():
    return jsonify({
        "height": world_state.shape[0],
        "width": world_state.shape[1],
        "channels": world_state.shape[2]
    })


@app.route("/stream")
def stream():
    global stopped
    if not (stopped):
        auto.step()
    auto.draw()
    world_state = auto.worldmap  # Should be (H,W,3) numpy array            
    world_state = world_state.reshape((world_state.shape[0]*world_state.shape[1], 3))
    data = compress_array(world_state)
    r = make_response(data)    
    setattr(r, "mimetype", "application/octet-stream")
    return r

@app.route("/streamstate")
def stream_state():
    global stopped
    return jsonify({"state": stopped})

@app.route("/keypress", methods=["GET"])
def keypress():
    global stopped
    global auto
    key = request.args.get("key")    
    if key == "Space":
        stopped = not stopped
    
    if key == "KeyR":
        auto.reset()

    return jsonify({"status": "success", "key": key})

if __name__ == "__main__":
    
    auto.step() # step the automaton
    auto.draw() # draw the worldstate
    world_state = auto.worldmap
    app.run(debug=True, host='0.0.0.0', port=5000)