import pygame
import pygame_gui
from pygame_gui.elements import UIHorizontalSlider, UIButton,UIWindow

pygame.init()

screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption('Slider Anchor Bug: set_anchors() AFTER creation')
clock = pygame.time.Clock()

manager = pygame_gui.UIManager((800, 600))

container = UIWindow(
    rect=pygame.Rect(50, 50, 700, 500),
    manager=manager
)

# Reference button to anchor to
reference_button = UIButton(
    relative_rect=pygame.Rect(50, 100, 150, 30),
    text="Anchor Target",
    manager=manager,
    container=container
)

slider = UIHorizontalSlider(
    relative_rect=pygame.Rect(60, 50, 300, 30),
    start_value=50,
    value_range=(0, 100),
    manager=manager,
    container=container,
)
anchors_dict = {
    'top': 'top',
    'top_target': reference_button,
    'centerx': 'centerx',
}
slider.set_anchors(anchors_dict)
# ================== IF THIS IS INCLUDE BEFORE set_relative_position, IT WORKS ==================
# slider.set_relative_position((60, 80))
if slider.button_container:
    slider.button_container.set_anchors(anchors_dict)
slider.set_relative_position((0,0))

running = True
while running:
    time_delta = clock.tick(60)/1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

        manager.process_events(event)

    manager.update(time_delta)

    screen.fill((50, 50, 50))

    manager.draw_ui(screen)

    pygame.display.update()

pygame.quit()
