import os, time, json
from flask import Flask, Response, jsonify
from flask_cors import CORS
import numpy as np

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

# Automaton world size 
W, H = 30, 30
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
auto = automaton_options[initial_automaton](H, W)
auto.step() # step the automaton
auto.draw() # draw the worldstate
world_state = auto.worldmap

app = Flask(__name__)
CORS(app, 
    origins=["http://localhost:5173"],
    allow_headers=["Content-Type", "Access-Control-Allow-Headers"],
    expose_headers=["Content-Type"],
    methods=["GET", "POST", "OPTIONS"],
)   

@app.route("/stream")
def stream():
    def generate():
        while True: 
            auto.step()
            auto.draw()
            world_state = auto.worldmap  # Should be (H,W,3) numpy array            
            if world_state.dtype != np.uint8:
                world_state = (world_state * 255).astype(np.uint8)

            # reshape the world state for three.js compatibility
            
            height = world_state.shape[0]
            width = world_state.shape[1]
            channels = world_state.shape[2]
            
            world_state = world_state.reshape((world_state.shape[0]*world_state.shape[1], 3))
            
            frame_data = {
                "frame": world_state.tolist(),
                "dimensions": {
                    "height": height,
                    "width": width,
                    "channels": channels
                },
                "timestamp": time.time()
            }
            
            yield f"data: {json.dumps(frame_data)}\n\n"
    
    response = Response(generate(), mimetype='text/event-stream')
    return response

@app.route("/hello", methods=["GET"])
def hello():
    return jsonify({"text": "Hello, World!"})

if __name__ == "__main__":

    app.run(debug=True, host='0.0.0.0', port=5000)