class Drawer:
    def __init__(self, FONT, HEIGHT, WIDTH,):
        self.FONT = FONT
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH

    def draw_text(self,frame, text, x, y, color=(255, 255, 255), scale=1):
        x_offset = 0
        for char in text:
            if char in self.FONT:
                glyph: list[list[int]] = self.FONT[char]
                for row_idx, row in enumerate(glyph):
                    for col_idx, pixel in enumerate(row):
                        if pixel:
                            for sy in range(scale):
                                for sx in range(scale):
                                    px: int = x + x_offset + (col_idx * scale) + sx
                                    py: int = y + (row_idx * scale) + sy
                                    if 0 <= px < self.WIDTH and 0 <= py < self.HEIGHT:
                                        frame[py, px] = color
                x_offset += 6 * scale
