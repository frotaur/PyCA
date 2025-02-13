import main
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run cellular automata simulation')
    
    parser.add_argument('-s', '--screen', nargs=2, type=int, default=(1280, 720),
                      help='Screen dimensions as width height (default: 1280 720)')
    
    parser.add_argument('-w', '--world', nargs=2, type=int, default=(250, 250),
                      help='World dimensions as width height (default: 250 250)')
    
    parser.add_argument('-d', '--device', type=str, default='cpu',
                      help='Device to run on: "cuda" or "cpu" (default: cpu)')

    args = parser.parse_args()
    return tuple(args.screen), tuple(args.world), args.device

if __name__ == '__main__':
    screen, world, device = parse_args()
    main.gameloop(screen, world, device)
