# PyCA
PyCA is a small library that facilitates the implementation of Artificial Life models (such as cellular automata) using tensor libraries, such as PyTorch.

All you need to do is sub-class the `Automaton` class, define how to `step` your model, define how to `draw` it, and you can visualize it in real time! 

## Installation
You need python 3.11 or later (might work with earlier versions). You can start using PyCA using `uv` (recommended) or `pip`, see instructions below.

### Recommended : using uv
We recommend using `uv` to install PyCA. You can install following [this link](https://docs.astral.sh/uv/getting-started/installation/).

From the root of the project, run:
```
uv sync
```
All set!

<font color="red">ONLY FOR WINDOWS PC WITH GPUS</font>:  If you are running windows, and have a NVIDIA GPU, run the following command before proceeding : `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --upgrade`. If you get cuda-related crashes afterwards, visit https://pytorch.org/ and try installing another version of `torch`.  


### Alternative: using pip
You can also install PyCA using pip. From the root of the project (ideally after activating a virtual environment), run:
```
pip install -e .
```
All set!
(Windows users, see note above about torch installation.)

## Run the implemented automatons
To run the main user interface that allows you to interact with the implementend automata, run : 

With `uv`:
```[python]
uv run python simulate.py
```

If you have a cuda GPU, run : 
```
uv run python simulate.py -d cuda
```

With `pip`, first make sure the venv in which you installe PyCA is activated, then run the same commands as above but without `uv run` at the beginning.

You can change the Screen size with the options `simulate.py [-h] [-s SCREEN SCREEN] [-w WORLD WORLD] [-d DEVICE]`
```options:
  -s WIDTH HEIGHT, --screen WIDTH HEIGHT
                        Screen dimensions as width height (default: 1280 720)
  -w WIDTH HEIGHT, --world WIDTH HEIGHT
                        World dimensions as width height (default: 250 250)
  -d DEVICE, --device DEVICE
                        Device to run on: "cuda" or "cpu" or "mps" (default: cpu)
```


## Tutorial
To learn to use the library, the best way is to follow the <a href='https://amldworlds.notion.site/PyCA-hands-on-199e18ef6eec806ea445f4e9a09edcee?pvs=74'>tutorial at this link.</a>. It will teach all the basics how to implement the Game of Life, with mouse and keyboard interactivity!

## Documentation
Documentation is under construction. In the meantime, the code is heavily documented with docstrings

## Code structure

```python
├─pyca
│  ├──automata
│  │   ├──models/ # All implemented automata
│  │   ├──utils/
│  │   ├──automaton.py # Base Automaton class
│  ├──interface/ # Utils for the PyCA GUI
├─extras/ # Currently only holding codebase to train Neurall Cellular Automata
├─demo_data # Where some saved parameters/states of automata are stored
├─simulate.py # Main entry script for PyCA  
```