# Skeleton for 1D CA, and more general cellular automata

This little project contains the necessary tools to easily implement and Visualize cellular automata, using python and pygame.
## How to use, quick
You need python 3.9 or later.
First install the dependencies with `pip install -r requirements.txt` or `pip3 install -r requirements.txt`. 
Run `python main.py` or `python3 main.py` and a window should open, displaying a 1D CA.

CAUTION ! : if using a windows PC, and you have a NVIDIA GPU, before running `pip install -r requirements.txt`, run `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` (or whatever is appropriate for your case, check pytorch's website). If you don't do this, python will install torch WITHOUT CUDA, and you won't be able to run your CA's on the GPU, making them much slower !
### Keyboard and mouse controls
- Camera
    - You can Zoom in and out by CTRL+mousewheel
    - You can move the camera around when zoom by holding CTRL and dragging with left click

- Keyboard commands
    - 'r' : start/stop recording. A red circle with appear, showing it is recording. Press 'R' again to stop. Video will be saved in  `./Videos/`
    - 'p' : save a picture of the current world. Saved in './Images'
    - 'spacebar' : start/stop automaton evolution.
    - 'q' : quits the program
    - 'DEL' : Resets the configuration
    - 'n' : Switch to a new rule



## Main
Main.py is in charge of displaying the CA dynamics, and interacting with them. The code is commented, so I suggest to take a look to see how it can be modified.
To play around with the default 1D CA, you can change the initial rule by modifying the 'wolfram_num' parameter to be any integer between 0 and 255. (see `auto = CA1D((H,W), wolfram_num=15, init_state=init_state)`).

To customize `main.py` :
- You can change the automaton to be some other version you implemented by setting the variable `auto`. See '`Automaton.py`' section for more info on implementing your own.
- You can change the possible interactions by modifying the code inside `for event in pygame.event.get():`.
    - Add other keyboard interactions by adding somethine inside the `if event.type == pygame.KEYDOWN :` context. The code inside ``` if(event.key == pygame.K_y): ``` will execute whenever you press the key 'y'.
    - There are prepared context that activate whenever the left/right mouse are clicked, and whenever we drag with this click. This can be used to draw inside the cellular automaton, for example.

Best way to understand how they work is to simply take a look at the code.

## Automaton
Inside `Automaton.py`, you will find two classes, `Automaton` and `CA1D`, which is a sub-class of `Automaton`. The docstring are quite extensive, so the code should be understandable from those only. To summarize :

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

- `CA1D`
    - This is an example representation of the Elementary Cellular automata of Wolfram. You initialize it with the size of the world, the wolfram number for the rule you want to simulate, and an optional initial state of the world (tensor (W,)).
    - Take a look at the implementation. Ideally, you want to avoid for-loops in space for cellular automata, to exploit to the maximum the parallelizability to run them fast on GPUs
    - In this case, the bottleneck is by far the Pygame visualization. If you rune the automaton in a separate script, not showing on screen and not calling 'draw()', it will be much faster.


## Alternatives

This skeleton code is very useful to design 2D CAs, and get an immediate and easy visualization. However, the two most important drawbacks are that pygame is very slow; you will not be able to simulate and view CAs faster than ~100 fps. Second drawback is that it is limited to 2D.

Other popular alternatives to make and view CAs are with game Engines, in particular Unity or Godot. The visualization capabilities are very advanced, and it is much easier to make 'good-looking' videos, but there is a trade-off in that it is much harder to implement the CA dynamics in parallel. This is done using Shaders/Compute Shaders, which are simply programs meant to run on Graphics Card, in a highly parallelized way. 

Since shaders were initially developed for Video Game Graphics, there is a long list of convetions which are adapted for this job. So the learning curve to start coding shaders is steeper and less intuitive than doing them in numpy, even though in the end the code is approximately the same.

Another option is WebGL, which is again some sort of shader language, but which can be exectued in the browser. It has the nice side effect that it allows one to very easily make a web-app of the automaton, which is nice for distribution. Take a look also at the [SwissGL library](https://github.com/google/swissgl), which is a way to simplify the WebGL code and essentially avoid a lot of boilerplate. 