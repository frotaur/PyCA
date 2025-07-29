"""
    Just a DEFAULTS dictionary with optional default parameters for each automaton.
    When an automaton is loaded, these parameters will be added in addition to the
    required (h,w) size, and device parameters.
"""


DEFAULTS = {
    "MultiLenia" : {
        "param_path": "lenia_cool_params"
    },
    "NeuralCA" : {
        "models_folder": 'saved_models/NCA'
    },
    "VonNeumann" : {
        "element_size": 9
    },
    "TotalisticCA1D" : {
        "rule": 1203,
        "k": 3,
        "r": 3
    },
}