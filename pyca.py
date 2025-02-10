import main

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run cellular automata simulation')
    
    parser.add_argument('-s', '--screen', nargs=2, type=int, default=(1280, 720),
                      help='Screen dimensions as width height (default: 800 600)')
    
    parser.add_argument('-w', '--world', nargs=2, type=int, default=(350, 350),
                      help='World dimensions as width height (default: 200 200)')
    
    parser.add_argument('-d', '--device', type=str, default='cuda',
                      help='Device to run on: "cuda" or "cpu" (default: cuda)')

    args = parser.parse_args()
    return tuple(args.screen), tuple(args.world), args.device

if __name__ == '__main__':
    screen, world, device = parse_args()
    main.gameloop(screen, world, device)
