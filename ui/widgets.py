import pygame

class Button:
    def __init__(self, rect, text, callback, tooltip=""):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.tooltip = tooltip
        self.hovered = False

    def draw(self, surface, font):
        color = (200, 200, 200) if self.hovered else (150, 150, 150)
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, (0, 0, 0), self.rect, 2)
        text_surf = font.render(self.text, True, (0, 0, 0))
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
        if self.tooltip:
            info_font = pygame.font.SysFont(None, 20)
            info_surf = info_font.render("ⓘ", True, (0, 0, 255))
            surface.blit(info_surf, (self.rect.right - 20, self.rect.top))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.hovered:
                self.callback()

class TextInput:
    def __init__(self, rect, default_text="", tooltip=""):
        self.rect = pygame.Rect(rect)
        self.text = default_text
        self.active = False
        self.tooltip = tooltip

    def draw(self, surface, font):
        color = (255, 255, 255) if self.active else (200, 200, 200)
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, (0, 0, 0), self.rect, 2)
        text_surf = font.render(self.text, True, (0, 0, 0))
        surface.blit(text_surf, (self.rect.x + 5, self.rect.y + 5))
        if self.tooltip:
            info_font = pygame.font.SysFont(None, 20)
            info_surf = info_font.render("ⓘ", True, (0, 0, 255))
            surface.blit(info_surf, (self.rect.right - 20, self.rect.top))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                self.text += event.unicode

    def get_value(self, cast_func, default):
        try:
            return cast_func(self.text)
        except Exception:
            return default

