import pygame
pygame.init()
pygame.font.init()  # Ensure font module is initialized

from ui import menu

def main():
    menu.main_menu()

if __name__ == '__main__':
    main()

