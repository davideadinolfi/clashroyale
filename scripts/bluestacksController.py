import pydirectinput as pdi
import time

class BlueStacksController:
    def __init__(self):
        # Coordinate carte in mano (calibra per il tuo schermo)
        self.card_positions = {
            1: (130, 887),  # slot 1
            2: (260, 887),  # slot 2
            3: (390, 887),  # slot 3
            4: (520, 887)  # slot 4
        }

        # Zona arena (y < 800 circa)
        self.arena_bounds = {
            'min_x': 100,
            'max_x': 620,
            'min_y': 200,
            'max_y': 800
        }