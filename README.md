# PyCA
PyCA is a small library that facilitates the implementation of Artificial Life models (such as cellular automata) using tensor libraries, such as PyTorch.

All you need to do is sub-class the `Automaton` class, define how to `step` your model, define how to `draw` it, and you can visualize it in real time! 
## Installation
You need python 3.11 or later (ideally, might work with earlier versions).

<font color="red"> FOR WINDOWS PC WITH GPUS :  If you are running windows, and have a NVIDIA GPU, please run the following command before proceeding : `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`. If something doesn't work afterwards, visit https://pytorch.org/ and choose an earlier version.  Linux and Mac users can safely skip this step.</font>


Install the pyca package by running `pip install -e .` from the projects directory (i.e., the same directory as `README.md`). You are all set!

## Run the implemented automatons
To run the main user interface that allows you to interact with the implementend automata, run : 
```[python]
python simulate.py
```

If you have a cuda GPU, run : 
```
python simulate.py -d cuda
```

More generally you can change the Screen size with the options `simulate.py [-h] [-s SCREEN SCREEN] [-w WORLD WORLD] [-d DEVICE]`
```options:
  -s SCREEN SCREEN, --screen SCREEN SCREEN
                        Screen dimensions as width height (default: 1280 720)
  -w WORLD WORLD, --world WORLD WORLD
                        World dimensions as width height (default: 200 200)
  -d DEVICE, --device DEVICE
                        Device to run on: "cuda" or "cpu" or "mps" (default: cpu)
```

<font color="red"> NOTE : 'mps' device is known to behave very differently. Other devices untested </font>


## Tutorial
To learn to use the library, the best way is to follow the <a href='https://amldworlds.notion.site/'>tutorial at this link.</a> It will teach all the basics how to implement the Game of Life, with mouse and keyboard interactivity!

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
├─saved_models # Where pre-trained/saved models are stored
├─lenia_cool_params # Cool parameters for Lenia automaton, to be moved to saved_models
├─train_nca/ # Codebase to train Neurall Cellular Automata
├─main.py # Logic for main Pygame loop
├─simulate.py # Main entry script for PyCA  
```

(Still under construction, will change in the future.)