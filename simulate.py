from pyca.interface.MainWindow import MainWindow
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run cellular automata simulation')
    
    parser.add_argument('-s', '--screen', nargs=2, type=int, default=(720, 1280),
                      help='Screen dimensions as height width (default: 720 1280)')

    parser.add_argument('-w', '--world', nargs=2, type=int, default=(250, 250),
                      help='World dimensions as height width (default: 250 250)')

    parser.add_argument('-d', '--device', type=str, default='cpu',
                      help='Device to run on: "cuda" or "cpu" (default: cpu)')

    parser.add_argument('-t', '--tablet_mode', action='store_true',
                      help='Enable tablet mode (default: False)')
    args = parser.parse_args()
    return tuple(args.screen), tuple(args.world), args.device, args.tablet_mode

if __name__ == '__main__':
    screen, world, device, tablet_mode = parse_args()
    window = MainWindow(screen, world, device, tablet_mode=tablet_mode)

    window.main_loop()
    