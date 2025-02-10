# Skeleton for 1D CA, and more general cellular automata

This little project contains the necessary tools to easily implement and Visualize cellular automata, using python and pygame.
## Installation
You need python 3.11 or later (ideally, might work with earlier versions).

<font color="red"> FOR WINDOWS PC WITH GPUS :  If you are running windows, and have a NVIDIA GPU, please run the following command before proceeding : `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`. If something doesn't work afterwards, visit https://pytorch.org/ and choose an earlier version.  Linux and Max users can safely skip this step.</font>


Install the pyca package by running `pip install -e .` from the projects directory (i.e., the same directory as `README.md`). You are all set!

## Run the implemented automatons
To run the main user interface that allows you to interact with the implementend automata you run the script 'simulate.py'.

The basic command is : `simulate.py [-h] [-s SCREEN SCREEN] [-w WORLD WORLD] [-d DEVICE]`
Options are as follows : 

```options:
  -s SCREEN SCREEN, --screen SCREEN SCREEN
                        Screen dimensions as width height (default: 1280 720)
  -w WORLD WORLD, --world WORLD WORLD
                        World dimensions as width height (default: 200 200)
  -d DEVICE, --device DEVICE
                        Device to run on: "cuda" or "cpu" or "mps" (default: cuda)
```
The only really important parameter is `-d DEVICE`, as the others can be changed in-simulation.

If you have a NVIDIA GPU, you should run it with `-d cuda`. If you have no GPU, you can run with `-d cpu`, it will be a little bit slower, so stick with small worlds. With a mac, you can try `-d mps`, which will be faster but pytorch is notoriously unreliable with mps, so some things might work differently (.e.g, for Lenia).

The 'screen ' parameter  determines the screen size of the window. The default parameter should be fine, but you can make it bigger if you want it to fill your screen. You can also directly modify the defaults in `simulate.py`.

The 'world' parameter determines the size of the simulated world. This can also be changed directly in the GUI, so mostly you won't need to change this parameter, except maybe the default value (again, in `simulate.py`) if you find yourself changing it every time.

## Automaton
Inside `pyca/automaton.py`, you will find 'Automaton', the base class for all artificial life models you will implement.The docstring are quite extensive, so the code should be understandable from those only. To summarize :

- Automaton
    - Base class for any Cellular Automaton 
    - Should be sub-classed when designing a new cellular automaton
    - It contains the following pre-existing attributes
        - `self._worldmap` : a (3,H,W) float tensor (i.e., an RGB image), that contains a representation of the current state of the cellular automaton
        - `self.size` : 2-uple (H,W) containing the height and width of the world
        - `self.h, self.w` : width and height of the world
        - `self.worldmap` : Not to be confused with `self._worldmap`. By default, it is a numpy (W,H,3) array of 8 bit integers (0-255), that contains the representation of the world. It is basically the same as `self._worldmap`, but translated for pygame.
    - The methods that should be overriden are the following : 
        - `self.draw()` : This method should update `self._worldmap`to correctly represent the current state of the world.
        - `self.step()` : This method, when called, should step the automaton once. The way the automaton state is represented internally is totally free.
