from capture import Capture
from ultralytics import YOLO
import easyocr
from scripts.yolo_live import update
import cv2
import numpy as np


def main():
    cap = Capture("BlueStacks")
    model = YOLO("runs/detect/yolo10dataset3_150iterMEDIO/weights/best.pt")
    reader = easyocr.Reader(['en'], gpu=False)
    while True:
        grid, elixir, timer, hand = update(cap, model , reader)
        # OPZIONE 1: Griglia con legenda
        show_grid(grid, cell_size=10, window_name="Game Grid")

        # OPZIONE 2: Solo griglia (più veloce)
        # show_grid_simple(grid, cell_size=10)

        # Info aggiuntive (opzionale)
        print(f"Elixir: {elixir}, Timer: {timer}, Hand: {hand}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def draw_grid(grid_matrix, cell_size=8):
    """
    Disegna la matrice griglia 72x128.

    Args:
        grid_matrix: matrice 72x128 con nomi oggetti o None
        cell_size: dimensione pixel di ogni cella

    Returns:
        Immagine numpy array
    """
    grid_rows = len(grid_matrix)
    grid_cols = len(grid_matrix[0])

    h = grid_rows * cell_size
    w = grid_cols * cell_size
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Mappa colori per oggetti
    color_map = {None: (40, 40, 40)}
    base_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 255), (255, 128, 0), (0, 128, 255), (128, 255, 0)
    ]
    color_idx = 0

    # Disegna celle
    for row in range(grid_rows):
        for col in range(grid_cols):
            obj_name = grid_matrix[row][col]

            # Assegna colore
            if obj_name not in color_map:
                color_map[obj_name] = base_colors[color_idx % len(base_colors)]
                color_idx += 1

            color = color_map[obj_name]

            y1 = row * cell_size
            y2 = (row + 1) * cell_size
            x1 = col * cell_size
            x2 = (col + 1) * cell_size

            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), 1)

    return img, color_map


def draw_legend(grid_matrix, color_map):
    """
    Crea legenda con colori e nomi oggetti.

    Args:
        grid_matrix: matrice 72x128
        color_map: dizionario {nome_oggetto: colore}

    Returns:
        Immagine legenda
    """
    # Trova oggetti unici
    unique_objects = set()
    for row in grid_matrix:
        for obj in row:
            if obj is not None:
                unique_objects.add(obj)

    if not unique_objects:
        return None

    legend_height = len(unique_objects) * 30 + 20
    legend_width = 200
    legend_img = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)

    y = 20
    for obj_name in sorted(unique_objects):
        color = color_map.get(obj_name, (255, 255, 255))
        cv2.rectangle(legend_img, (10, y - 10), (30, y + 10), color, -1)
        cv2.putText(legend_img, obj_name, (40, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 30

    return legend_img


def show_grid(grid_matrix, cell_size=10, window_name="Grid View"):
    """
    Mostra la griglia con legenda.

    Args:
        grid_matrix: matrice 72x128
        cell_size: dimensione pixel cella
        window_name: nome finestra
    """
    grid_img, color_map = draw_grid(grid_matrix, cell_size)
    legend_img = draw_legend(grid_matrix, color_map)

    if legend_img is not None:
        # Combina griglia + legenda
        h = max(grid_img.shape[0], legend_img.shape[0])
        combined = np.zeros((h, grid_img.shape[1] + legend_img.shape[1], 3), dtype=np.uint8)
        combined[:grid_img.shape[0], :grid_img.shape[1]] = grid_img
        combined[:legend_img.shape[0], grid_img.shape[1]:] = legend_img
        cv2.imshow(window_name, combined)
    else:
        cv2.imshow(window_name, grid_img)


def show_grid_simple(grid_matrix, cell_size=10, window_name="Grid View"):
    """
    Mostra solo la griglia senza legenda.

    Args:
        grid_matrix: matrice 72x128
        cell_size: dimensione pixel cella
        window_name: nome finestra
    """
    grid_img, _ = draw_grid(grid_matrix, cell_size)
    cv2.imshow(window_name, grid_img)

if __name__ == "__main__":
    main()




