import tkinter as tk
import threading
import time
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES NEURONALES
# ═══════════════════════════════════════════════════════════════════════════════
DT = 0.05
TAU_NR = 0.5
TAU_GA = 0.8
TAU_MR = 0.6
TAU_MEM = 2.0
TAU_RET = 1.5
TAU1_OSC = 0.3
TAU2_OSC = 0.8
A_OSC = 2.0
B_OSC = 0.5
C_OSC = 0.6
D_OSC = 0.4
W_EXCIT = 0.6
W_INHIB = 0.3
UMBRAL_RET = 0.65

# ═══════════════════════════════════════════════════════════════════════════════
# TAMAÑO DE CELDA RECTANGULAR  (proporcional a 15 cm × 11.8 cm reales)
# ═══════════════════════════════════════════════════════════════════════════════
CELL_W = 60   # ancho  en píxeles  (representa 15 cm)
CELL_H = 47   # alto   en píxeles  (representa 11.8 cm)

# ═══════════════════════════════════════════════════════════════════════════════
# DIRECCIONES Y MAPEOS
# ═══════════════════════════════════════════════════════════════════════════════
_DIRS4 = ["N", "E", "S", "O"]
_DIRS8 = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
DX = {"N": 0, "S": 0, "E": 1, "O": -1, "NE": 1, "SE": 1, "SO": -1, "NO": -1}
DY = {"N": -1, "S": 1, "E": 0, "O": 0, "NE": -1, "SE": 1, "SO": 1, "NO": -1}
OPP = {"N": "S", "S": "N", "E": "O", "O": "E", "NE": "SO", "SE": "NO", "SO": "NE", "NO": "SE"}
CARDINALS = {"N": 90, "E": 0, "S": 270, "O": 180}
CORNERS = {"NE": 45, "SE": 315, "SO": 225, "NO": 135}
DIR_VECTORS = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "O": (-1, 0)}

# Índices neuronales
I_NR = {d: i for i, d in enumerate(_DIRS4)}
I_GA = {d: i+4 for i, d in enumerate(_DIRS4)}
I_MR = {d: i+8 for i, d in enumerate(_DIRS8)}
I_ON = 16
I_OFF = 17
I_ADON = 18
I_ADOF = 19
I_MEM = 20
I_RET = 21

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE ACTIVACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
def naka_rushton_f(x, M=1.0, n=2.0, C=0.2):
    if x <= 0:
        return 0.0
    return M * (x**n) / (x**n + C**n)

def gaussian_f(theta1, theta2, sigma=45.0):
    diff = min(abs(theta1 - theta2), 360 - abs(theta1 - theta2))
    return math.exp(-(diff**2) / (2 * sigma**2))

# ═══════════════════════════════════════════════════════════════════════════════
# CONTROLADOR NEURONAL
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class CellInfo:
    pos: Tuple[int, int]
    tried: Set[str] = field(default_factory=set)
    visit_count: int = 0
    dead_end: bool = False
    exits_taken: Set[str] = field(default_factory=set)

class NeuroController:
    def __init__(self):
        self.z = [0.0] * 22
        self.s_prox = {"N": 0.0, "E": 0.0, "S": 0.0, "O": 0.0}
        self.open_dirs = []
        self.current_pos = (0, 0)
        self.naka_act = {"N": 0.0, "E": 0.0, "S": 0.0, "O": 0.0}
        self.gauss_act = {"N": 0.0, "E": 0.0, "S": 0.0, "O": 0.0}
        self.motor_ring = {d: 0.0 for d in _DIRS8}
        self.cell_info: Dict[Tuple[int, int], CellInfo] = {}
        self.visit_count: Dict[Tuple[int, int], int] = {}
        self.move_log = []
        self.recent_moves = deque(maxlen=12)
        self.last_move = None
        self.move_decision = None
        self.backtracking = False
        self.backtrack_target = None
        self.backtrack_path = []
        self.decision_stack = []
        self.checkpoint_active = False
        self.retroceso_permitido = False
        self.loop_detected = False
        self.no_progress_counter = 0
        self.last_position = None
        self.osc_on = 0.0
        self.osc_off = 0.0
        self.mem_signal = 0.0
        self.ret_signal = 0.0

    def update_sensors(self, distances: Dict[str, float]):
        max_dist = 400
        for d in _DIRS4:
            self.s_prox[d] = min(1.0, distances.get(d, 0) / max_dist)

    def set_open_dirs(self, open_dirs: List[str], pos: Tuple[int, int]):
        self.open_dirs = open_dirs
        self.current_pos = pos
        if pos not in self.cell_info:
            self.cell_info[pos] = CellInfo(pos=pos)
        self.cell_info[pos].visit_count += 1
        self.visit_count[pos] = self.cell_info[pos].visit_count
        if len(open_dirs) == 1 and self.cell_info[pos].visit_count > 1:
            self.cell_info[pos].dead_end = True

    def _needs_backtrack(self) -> bool:
        if self.backtracking:
            return True
        if len(self.open_dirs) == 0:
            return True
        current_visits = self.visit_count.get(self.current_pos, 0)
        if current_visits > 3:
            cell = self.cell_info.get(self.current_pos)
            if cell:
                unexplored = [d for d in self.open_dirs if d not in cell.tried]
                if not unexplored:
                    return True
        if len(self.recent_moves) >= 8:
            last_moves = list(self.recent_moves)[-6:]
            pattern = "".join(last_moves)
            if "NSNS" in pattern or "SNSN" in pattern or "EOEO" in pattern or "OEOE" in pattern:
                self.loop_detected = True
                return True
        return False

    def _find_unexplored_path(self) -> Optional[Tuple[int, int, List[str]]]:
        visited = set()
        queue = deque([(self.current_pos, [])])
        visited.add(self.current_pos)
        while queue:
            pos, path = queue.popleft()
            x, y = pos
            cell = self.cell_info.get(pos)
            if cell:
                for d in _DIRS4:
                    if d in self.open_dirs and d not in cell.tried:
                        if not cell.dead_end:
                            return (x, y, path)
            for d in _DIRS4:
                nx, ny = x + DX[d], y + DY[d]
                new_pos = (nx, ny)
                if new_pos not in visited:
                    if new_pos in self.cell_info or (0 <= nx < 20 and 0 <= ny < 15):
                        visited.add(new_pos)
                        queue.append((new_pos, path + [d]))
        return None

    def _start_backtrack(self):
        if self.backtracking:
            return
        result = self._find_unexplored_path()
        if result:
            target_pos, path = result[0:2]
            self.backtracking = True
            self.backtrack_target = target_pos
            self.backtrack_path = path
            print(f"Backtrack iniciado hacia {target_pos}")
        else:
            self._reset_exploration()

    def _reset_exploration(self):
        for pos, cell in self.cell_info.items():
            if cell.visit_count > 2 and not cell.dead_end:
                cell.tried = set()
        self.backtracking = False
        self.loop_detected = False
        self.no_progress_counter = 0
        print("Reset parcial de exploración")

    def _execute_backtrack(self):
        if not self.backtrack_target:
            self.backtracking = False
            return
        tx, ty = self.backtrack_target
        cx, cy = self.current_pos
        dx = tx - cx
        dy = ty - cy
        if abs(dx) > abs(dy):
            target_dir = "E" if dx > 0 else "O"
        else:
            target_dir = "S" if dy > 0 else "N"
        if target_dir in self.open_dirs:
            self.move_decision = target_dir
            return
        else:
            for d in self.open_dirs:
                if d != OPP.get(self.last_move):
                    self.move_decision = d
                    return
        if self.current_pos == self.backtrack_target:
            self.backtracking = False
            self.backtrack_target = None
            self.backtrack_path = []

    def _calculate_motor_ring(self):
        motor_vals = {}
        for d in _DIRS4:
            base_act = self.gauss_act[d]
            cell = self.cell_info.get(self.current_pos)
            if cell and d in cell.tried:
                base_act *= 0.3
            nx, ny = self.current_pos[0] + DX[d], self.current_pos[1] + DY[d]
            neighbor = self.cell_info.get((nx, ny))
            if neighbor and neighbor.dead_end:
                base_act *= 0.1
            if neighbor and neighbor.visit_count > 2:
                base_act *= max(0.2, 1.0 - neighbor.visit_count * 0.2)
            motor_vals[d] = base_act
        for corner, (d1, d2) in [("NE", ("N", "E")), ("SE", ("S", "E")),
                                   ("SO", ("S", "O")), ("NO", ("N", "O"))]:
            motor_vals[corner] = math.sqrt(motor_vals[d1] * motor_vals[d2])
        for d in _DIRS8:
            self.motor_ring[d] = motor_vals.get(d, 0.0)

    def _choose_direction(self) -> Optional[str]:
        if len(self.move_log) == 0:
            if self.open_dirs:
                best_dir = max(self.open_dirs, key=lambda d: self.motor_ring[d])
                return best_dir
            return None
        cell = self.cell_info.get(self.current_pos)
        unexplored = []
        if cell:
            unexplored = [d for d in self.open_dirs if d not in cell.tried]
        back_dir = OPP.get(self.last_move) if self.last_move else None
        if unexplored:
            valid_dirs = [d for d in unexplored if d != back_dir]
            if valid_dirs:
                best_dir = max(valid_dirs, key=lambda d: self.motor_ring[d])
            else:
                best_dir = max(unexplored, key=lambda d: self.motor_ring[d])
            return best_dir
        elif self.open_dirs:
            valid_dirs = [d for d in self.open_dirs if d != back_dir]
            if valid_dirs:
                best_dir = max(valid_dirs, key=lambda d: self.motor_ring[d])
            else:
                best_dir = max(self.open_dirs, key=lambda d: self.motor_ring[d])
            return best_dir
        return None

    def step(self, dt=DT):
        z = self.z
        for d, idx in I_NR.items():
            f = naka_rushton_f(self.s_prox[d])
            z[idx] = z[idx] + (dt/TAU_NR)*(-z[idx] + f)
            z[idx] = max(0.0, min(1.0, z[idx]))
            self.naka_act[d] = float(z[idx])
        nr = {d: float(z[I_NR[d]]) for d in _DIRS4}
        for i, d in enumerate(_DIRS4):
            dl = _DIRS4[(i-1) % 4]
            dr = _DIRS4[(i+1) % 4]
            do = _DIRS4[(i+2) % 4]
            u = (gaussian_f(CARDINALS[d], CARDINALS[d]) * nr[d]
               + W_EXCIT * gaussian_f(CARDINALS[d], CARDINALS[dl]) * nr[dl]
               + W_EXCIT * gaussian_f(CARDINALS[d], CARDINALS[dr]) * nr[dr]
               - W_INHIB * gaussian_f(CARDINALS[d], CARDINALS[do]) * nr[do])
            u = max(0.0, min(1.0, u))
            z[I_GA[d]] = z[I_GA[d]] + (dt/TAU_GA)*(-z[I_GA[d]] + u)
            z[I_GA[d]] = max(0.0, min(1.0, z[I_GA[d]]))
            self.gauss_act[d] = float(z[I_GA[d]])
        self._calculate_motor_ring()
        for d in _DIRS8:
            idx = I_MR[d]
            z[idx] = z[idx] + (dt/TAU_MR)*(-z[idx] + self.motor_ring[d])
            z[idx] = max(0.0, min(1.0, z[idx]))
        K1 = float(z[I_MR["NE"]]) + float(z[I_MR["NO"]])
        K2 = float(z[I_MR["SE"]]) + float(z[I_MR["SO"]])
        zn = (K1+K2)/2.0 + 1e-6
        u_on = max(0.0, zn*K1 - D_OSC*float(z[I_OFF]))
        u_off = max(0.0, zn*K2 - D_OSC*float(z[I_ON]))
        z[I_ON] += (dt/TAU1_OSC)*(-z[I_ON] + A_OSC*u_on**2 / ((B_OSC+z[I_ADON])**2 + u_on**2 + 1e-12))
        z[I_OFF] += (dt/TAU1_OSC)*(-z[I_OFF] + A_OSC*u_off**2 / ((B_OSC+z[I_ADOF])**2 + u_off**2 + 1e-12))
        z[I_ADON] += (dt/TAU2_OSC)*(-z[I_ADON] + C_OSC*z[I_ON])
        z[I_ADOF] += (dt/TAU2_OSC)*(-z[I_ADOF] + C_OSC*z[I_OFF])
        for idx in [I_ON, I_OFF, I_ADON, I_ADOF]:
            z[idx] = max(0.0, z[idx])
        self.osc_on = float(z[I_ON])
        self.osc_off = float(z[I_OFF])
        back = OPP.get(self.last_move) if self.last_move else None
        fwd = [d for d in self.open_dirs if d != back]
        loop_penalty = 1.0 if self.loop_detected else 0.0
        dead = 1.0 if (not fwd and not self.backtracking) else 0.0
        mem_input = max(dead, loop_penalty)
        z[I_MEM] = z[I_MEM] + (dt/TAU_MEM)*(-z[I_MEM] + mem_input)
        z[I_MEM] = max(0.0, min(1.0, z[I_MEM]))
        self.mem_signal = float(z[I_MEM])
        ret = 1.0 if self._needs_backtrack() else 0.0
        z[I_RET] = z[I_RET] + (dt/TAU_RET)*(-z[I_RET] + ret)
        z[I_RET] = max(0.0, min(1.0, z[I_RET]))
        self.ret_signal = float(z[I_RET])
        self.retroceso_permitido = bool(z[I_RET] > UMBRAL_RET)
        if self.backtracking:
            self._execute_backtrack()
        elif self.retroceso_permitido and not self.open_dirs:
            self._start_backtrack()
        else:
            if len(self.move_log) == 0:
                self.backtracking = False
                self.retroceso_permitido = False
            self.move_decision = self._choose_direction()
        if self.current_pos == self.last_position:
            self.no_progress_counter += 1
        else:
            self.no_progress_counter = 0
            self.last_position = self.current_pos
        if self.no_progress_counter > 20:
            self.loop_detected = True
            self._start_backtrack()
            self.no_progress_counter = 0


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN VISUAL
# ═══════════════════════════════════════════════════════════════════════════════
BG        = "#0a0e1a"
PANEL     = "#0f1628"
ACCENT    = "#00d4ff"
ACCENT2   = "#ff6b35"
GREEN     = "#39ff14"
YELLOW    = "#ffd700"
MAGENTA   = "#ff00aa"
GRAY      = "#1e2840"
DARK_GRAY = "#141c2e"
TEXT      = "#c8d8f0"
WHITE     = "#ffffff"
CYAN2     = "#00ffcc"

# Laberinto manual
MI_LABERINTO = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
]

def generate_maze(cols, rows, complexity=0.4):
    rows = len(MI_LABERINTO)
    cols = len(MI_LABERINTO[0])
    walls_h = [[True]*cols for _ in range(rows+1)]
    walls_v = [[True]*(cols+1) for _ in range(rows)]
    for y in range(rows):
        for x in range(cols):
            if MI_LABERINTO[y][x] == 0:
                if x < cols - 1 and MI_LABERINTO[y][x+1] == 0:
                    walls_v[y][x+1] = False
                if y < rows - 1 and MI_LABERINTO[y+1][x] == 0:
                    walls_h[y+1][x] = False
    return walls_h, walls_v


# ═══════════════════════════════════════════════════════════════════════════════
# ROBOT  (adaptado para celdas rectangulares)
# ═══════════════════════════════════════════════════════════════════════════════
class MazeRobot:
    def __init__(self, walls_h, walls_v, cols, rows, cell_w, cell_h):
        self.walls_h = walls_h
        self.walls_v = walls_v
        self.cols = cols
        self.rows = rows
        self.cell_w = cell_w   # ancho de celda en px
        self.cell_h = cell_h   # alto  de celda en px
        self.nc = NeuroController()

        sx, sy = 1, 7          # celda de inicio
        gx_cell, gy_cell = 8, 1   # celda meta

        self.x = sx * cell_w + cell_w // 2
        self.y = sy * cell_h + cell_h // 2
        self.gx = gx_cell * cell_w + cell_w // 2
        self.gy = gy_cell * cell_h + cell_h // 2

        self.path = [(self.x, self.y)]
        self.step_count = 0
        self.solved = False

    def get_cell(self):
        return int(self.x // self.cell_w), int(self.y // self.cell_h)

    def can_move(self, col, row, d):
        if d == "N": return row > 0           and not self.walls_h[row][col]
        if d == "S": return row < self.rows-1 and not self.walls_h[row+1][col]
        if d == "E": return col < self.cols-1 and not self.walls_v[row][col+1]
        if d == "O": return col > 0           and not self.walls_v[row][col]
        return False

    def cast_ray(self, direction):
        col, row = self.get_cell()
        c, r, dist = col, row, 0
        while dist < 10:
            if not self.can_move(c, r, direction):
                break
            c += DX[direction]
            r += DY[direction]
            dist += 1
        # Distancia en píxeles: horizontal usa cell_w, vertical usa cell_h
        if direction in ("E", "O"):
            return dist * self.cell_w
        else:
            return dist * self.cell_h

    def sense(self):
        distances, open_dirs = {}, []
        col, row = self.get_cell()
        for d in _DIRS4:
            distances[d] = self.cast_ray(d)
            if self.can_move(col, row, d):
                open_dirs.append(d)
        self.nc.update_sensors(distances)
        self.nc.set_open_dirs(open_dirs, (col, row))

    def move(self):
        dec = self.nc.move_decision
        if dec is None:
            return
        col, row = self.get_cell()
        if not self.can_move(col, row, dec):
            self.nc.move_decision = None
            return

        origin = (col, row)

        if dec == "N": row -= 1
        elif dec == "S": row += 1
        elif dec == "E": col += 1
        elif dec == "O": col -= 1

        self.x = col * self.cell_w + self.cell_w // 2
        self.y = row * self.cell_h + self.cell_h // 2
        self.path.append((self.x, self.y))
        self.step_count += 1

        info = self.nc.cell_info.get(origin)
        if info:
            info.tried.add(dec)
            info.exits_taken.add(dec)

        self.nc.last_move = dec
        self.nc.move_log.append(dec)
        self.nc.recent_moves.append(dec)
        if len(self.nc.recent_moves) > 12:
            self.nc.recent_moves.popleft()
        self.nc.move_decision = None

        if col == self.gx // self.cell_w and row == self.gy // self.cell_h:
            self.solved = True


# ═══════════════════════════════════════════════════════════════════════════════
# INTERFAZ GRÁFICA
# ═══════════════════════════════════════════════════════════════════════════════
class NeuroMazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuroControl Maze — Celdas Rectangulares (15×11.8 cm)")
        self.root.configure(bg=BG)
        self.root.geometry("1400x900")
        self.running = False
        self.sim_thread = None

        self.maze_rows = len(MI_LABERINTO)
        self.maze_cols = len(MI_LABERINTO[0])

        self.robot = None
        self._build_ui()
        self.new_maze()

    def _build_ui(self):
        hdr = tk.Frame(self.root, bg=BG, pady=6)
        hdr.pack(fill="x")
        tk.Label(hdr, text="NEUROCONTROL MAZE — CELDAS 15×11.8 cm",
                 font=("Courier New", 16, "bold"), fg=ACCENT2, bg=BG).pack(side="left", padx=18)
        tk.Label(hdr,
                 text="Naka-Rushton · Gaussiana · Wilson-Cowan · Anti-bucle · Memoria DFS",
                 font=("Courier New", 9), fg=TEXT, bg=BG).pack(side="left", padx=4)

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=10, pady=4)

        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        tk.Label(left, text="LABERINTO", font=("Courier New", 9, "bold"),
                 fg=ACCENT2, bg=BG).pack(anchor="w", padx=6)

        # Canvas con dimensiones rectangulares
        self.maze_canvas = tk.Canvas(left, bg=DARK_GRAY,
                                     highlightthickness=1, highlightbackground=ACCENT,
                                     width=self.maze_cols * CELL_W,
                                     height=self.maze_rows * CELL_H)
        self.maze_canvas.pack(padx=6, pady=4)

        ctrl = tk.Frame(left, bg=BG)
        ctrl.pack(fill="x", padx=6, pady=4)
        self._btn(ctrl, "RESETEAR", self.new_maze,  ACCENT).pack(side="left", padx=4)
        self._btn(ctrl, "INICIAR",  self.start_sim, GREEN).pack(side="left", padx=4)
        self._btn(ctrl, "DETENER",  self.stop_sim,  ACCENT2).pack(side="left", padx=4)

        mem_f = tk.LabelFrame(left, text=" MEMORIA ANTI-REPETICIÓN ",
                              font=("Courier New", 9, "bold"),
                              fg=MAGENTA, bg=PANEL, bd=1, relief="solid", labelanchor="n")
        mem_f.pack(fill="x", padx=6, pady=4)
        self.mem_label = tk.Label(mem_f, text="Movimientos: []",
                                  font=("Courier New", 9), fg=MAGENTA,
                                  bg=PANEL, wraplength=460, justify="left")
        self.mem_label.pack(padx=8, pady=3, anchor="w")
        self.cell_info_label = tk.Label(mem_f, text="Celdas exploradas: 0 | Repeticiones: 0",
                                        font=("Courier New", 9), fg=YELLOW, bg=PANEL)
        self.cell_info_label.pack(padx=8, pady=2, anchor="w")
        self.mem_signal_label = tk.Label(mem_f, text="z[20] mem_signal: 0.000",
                                         font=("Courier New", 9), fg=YELLOW, bg=PANEL)
        self.mem_signal_label.pack(padx=8, pady=2, anchor="w")

        c6_f = tk.LabelFrame(left, text=" RETROCESO NEURONAL (z[21]) ",
                             font=("Courier New", 9, "bold"),
                             fg=CYAN2, bg=PANEL, bd=1, relief="solid", labelanchor="n")
        c6_f.pack(fill="x", padx=6, pady=4)
        self.ret_signal_label = tk.Label(c6_f, text="z[21]: 0.000  |  Permiso: NO",
                                         font=("Courier New", 9, "bold"), fg=GRAY, bg=PANEL)
        self.ret_signal_label.pack(padx=8, pady=4, anchor="w")
        self.ret_bar_canvas = tk.Canvas(c6_f, bg=PANEL, height=14,
                                        highlightthickness=0, width=460)
        self.ret_bar_canvas.pack(padx=8, pady=(0,6), anchor="w")

        right = tk.Frame(main, bg=BG, width=420)
        right.pack(side="right", fill="y", padx=8)
        right.pack_propagate(False)

        tk.Label(right, text="ESTADO NEURONAL", font=("Courier New", 9, "bold"),
                 fg=ACCENT2, bg=BG).pack(anchor="w", padx=4)
        self.neuro_canvas = tk.Canvas(right, bg=DARK_GRAY, width=400, height=300,
                                      highlightthickness=1, highlightbackground=ACCENT)
        self.neuro_canvas.pack(padx=4, pady=4)

        dec_f = tk.LabelFrame(right, text=" DECISIÓN ACTUAL ",
                              font=("Courier New", 9, "bold"),
                              fg=ACCENT, bg=PANEL, bd=1, relief="solid", labelanchor="n")
        dec_f.pack(fill="x", padx=4, pady=4)
        self.decision_label = tk.Label(dec_f, text="-",
                                       font=("Courier New", 34, "bold"), fg=GREEN, bg=PANEL)
        self.decision_label.pack(pady=4)
        self.decision_detail = tk.Label(dec_f, text="Esperando inicio...",
                                        font=("Courier New", 9), fg=TEXT, bg=PANEL)
        self.decision_detail.pack(pady=2)

        bars_f = tk.LabelFrame(right, text=" ACTIVACIONES POR CAPA ",
                               font=("Courier New", 9, "bold"),
                               fg=ACCENT2, bg=PANEL, bd=1, relief="solid", labelanchor="n")
        bars_f.pack(fill="both", expand=True, padx=4, pady=4)
        self.bar_canvas = tk.Canvas(bars_f, bg=PANEL, highlightthickness=0)
        self.bar_canvas.pack(fill="both", expand=True, padx=4, pady=4)

        self.step_label = tk.Label(right, text="Pasos: 0  |  Estado: DETENIDO",
                                   font=("Courier New", 8), fg=TEXT, bg=BG)
        self.step_label.pack(anchor="w", padx=4, pady=2)

    def _btn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd,
                         font=("Courier New", 9, "bold"),
                         fg=BG, bg=color, activebackground=WHITE,
                         activeforeground=BG, bd=0, padx=10, pady=4,
                         cursor="hand2", relief="flat")

    def new_maze(self):
        self.stop_sim()
        time.sleep(0.12)
        wh, wv = generate_maze(self.maze_cols, self.maze_rows)
        self.robot = MazeRobot(wh, wv, self.maze_cols, self.maze_rows, CELL_W, CELL_H)
        self._draw_maze()
        self._update_neuro_display()
        self.decision_label.config(text="-", fg=GREEN)
        self.decision_detail.config(text="Listo. Presiona INICIAR")
        self.step_label.config(text="Pasos: 0  |  Estado: DETENIDO")

    def _draw_maze(self):
        c = self.maze_canvas
        c.delete("all")
        r = self.robot
        cw, ch = CELL_W, CELL_H

        # Fondo de celdas
        for row in range(self.maze_rows):
            for col in range(self.maze_cols):
                c.create_rectangle(col*cw, row*ch, col*cw+cw, row*ch+ch,
                                   fill=DARK_GRAY, outline="")

        # Paredes horizontales
        for row in range(self.maze_rows+1):
            for col in range(self.maze_cols):
                if r.walls_h[row][col]:
                    c.create_line(col*cw, row*ch, col*cw+cw, row*ch,
                                  fill=ACCENT, width=2)

        # Paredes verticales
        for row in range(self.maze_rows):
            for col in range(self.maze_cols+1):
                if r.walls_v[row][col]:
                    c.create_line(col*cw, row*ch, col*cw, row*ch+ch,
                                  fill=ACCENT, width=2)

        # Meta
        gx0, gy0 = r.gx - cw//2 + 4, r.gy - ch//2 + 4
        c.create_rectangle(gx0, gy0, gx0+cw-8, gy0+ch-8,
                           fill="", outline=YELLOW, width=2, dash=(4,3))
        c.create_text(r.gx, r.gy, text="X", fill=YELLOW,
                      font=("Courier New", 14, "bold"))

        self._draw_robot()

    def _draw_robot(self):
        c, r, nc = self.maze_canvas, self.robot, self.robot.nc
        c.delete("path")
        c.delete("robot")
        c.delete("sensor")
        cw, ch = CELL_W, CELL_H

        # Celdas visitadas
        for (x, y), count in nc.visit_count.items():
            intensity = max(40, 120 - count * 15)
            color = f"#{intensity:02x}{intensity:02x}{intensity+40:02x}"
            c.create_rectangle(x*cw, y*ch, x*cw+cw, y*ch+ch,
                               fill=color, outline="", tag="path")

        # Trayectoria
        if len(r.path) > 1:
            pts = [coord for px, py in r.path for coord in (px, py)]
            c.create_line(*pts, fill="#2a8a4a", width=2, smooth=True, tag="path")

        # Sensores láser
        s_colors = {"N": ACCENT, "E": ACCENT2, "S": MAGENTA, "O": YELLOW}
        for d, (dx, dy) in DIR_VECTORS.items():
            act = nc.naka_act[d]
            if act > 0.05:
                ln = max(4, int(nc.s_prox[d] * 60))
                c.create_line(r.x, r.y, r.x+dx*ln, r.y+dy*ln,
                              fill=s_colors[d], width=max(1, int(act*3)), tag="sensor")
                c.create_oval(r.x+dx*ln-3, r.y+dy*ln-3, r.x+dx*ln+3, r.y+dy*ln+3,
                              fill=s_colors[d], outline="", tag="sensor")

        rad = 8   # radio ligeramente menor para que quepa en celdas más pequeñas
        if nc.retroceso_permitido:
            c.create_oval(r.x-rad-6, r.y-rad-6, r.x+rad+6, r.y+rad+6,
                          fill="", outline=CYAN2, width=3, tag="robot")
        if nc.backtracking:
            c.create_oval(r.x-rad-2, r.y-rad-2, r.x+rad+2, r.y+rad+2,
                          fill="", outline=YELLOW, width=1, dash=(3,2), tag="robot")
        if nc.loop_detected:
            c.create_oval(r.x-rad-8, r.y-rad-8, r.x+rad+8, r.y+rad+8,
                          fill="", outline=MAGENTA, width=2, dash=(5,2), tag="robot")

        fill_col = CYAN2 if nc.retroceso_permitido else (MAGENTA if nc.backtracking else GREEN)
        c.create_oval(r.x-rad, r.y-rad, r.x+rad, r.y+rad,
                      fill=fill_col, outline=WHITE, width=1, tag="robot")

        dec = nc.move_decision
        if dec:
            dx, dy = DIR_VECTORS[dec]
            c.create_line(r.x, r.y, r.x+dx*12, r.y+dy*12,
                          fill=WHITE, width=2, arrow="last", arrowshape=(6,8,3), tag="robot")
            c.create_text(r.x, r.y, text=dec, fill=BG,
                          font=("Courier New", 7, "bold"), tag="robot")

    def _update_neuro_display(self):
        nc, c = self.robot.nc, self.neuro_canvas
        c.delete("all")
        cx, cy, R = 200, 150, 110
        c.create_oval(cx-R, cy-R, cx+R, cy+R, fill=BG, outline=GRAY, width=1)
        for pct in [0.33, 0.66, 1.0]:
            r2 = R*pct
            c.create_oval(cx-r2, cy-r2, cx+r2, cy+r2, fill="", outline="#2d3d5a", width=1)
        for ang in range(0, 360, 45):
            rad = math.radians(ang)
            c.create_line(cx, cy, cx+R*math.cos(rad), cy-R*math.sin(rad),
                          fill="#2d3d5a", width=1)
        pts = []
        for d, ang_deg in CARDINALS.items():
            ang_r = math.radians(ang_deg)
            r2 = nc.gauss_act[d] * R
            pts += [cx+r2*math.cos(ang_r), cy-r2*math.sin(ang_r)]
        if pts:
            c.create_polygon(*pts, fill="#0a3040", outline=ACCENT, width=1, smooth=True)
        for d, ang_deg in {**CARDINALS, **CORNERS}.items():
            ang_r = math.radians(ang_deg)
            act = nc.motor_ring[d]
            r2 = act * R
            bx, by = cx+r2*math.cos(ang_r), cy-r2*math.sin(ang_r)
            rd = 4 + int(act*6)
            if act > 0.08:
                c.create_oval(bx-rd, by-rd, bx+rd, by+rd,
                              fill=(ACCENT2 if d in CORNERS else ACCENT),
                              outline=WHITE, width=1)
        for d, ang_deg in CARDINALS.items():
            act = nc.naka_act[d]
            ang_r = math.radians(ang_deg)
            r_i, r_o = R+6, R+6+act*18
            c.create_line(cx+r_i*math.cos(ang_r), cy-r_i*math.sin(ang_r),
                          cx+r_o*math.cos(ang_r), cy-r_o*math.sin(ang_r),
                          fill=YELLOW, width=3)
        osc_r = 14 + nc.osc_on*10
        col_osc = MAGENTA if nc.osc_on > 0.3 else GRAY
        c.create_oval(cx-osc_r, cy-osc_r, cx+osc_r, cy+osc_r,
                      fill="", outline=col_osc, width=2)
        ret_r = 6 + nc.ret_signal * 10
        c6col = CYAN2 if nc.retroceso_permitido else GRAY
        c.create_oval(cx-ret_r+30, cy-ret_r-30, cx+ret_r+30, cy+ret_r-30,
                      fill="", outline=c6col, width=2, dash=(3,2))
        for d, ang_deg in {**CARDINALS, **CORNERS}.items():
            ang_r = math.radians(ang_deg)
            lx = cx+(R+20)*math.cos(ang_r)
            ly = cy-(R+20)*math.sin(ang_r)
            act = nc.motor_ring[d]
            col = GREEN if act > 0.5 else (ACCENT if d in CARDINALS else ACCENT2)
            c.create_text(lx, ly, text=d, fill=col,
                          font=("Courier New", 9 if d in CARDINALS else 7,
                                "bold" if d in CARDINALS else "normal"))
        c.create_text(8, 8,  text="- Gaussiana C2",   fill=ACCENT,  font=("Courier New",6), anchor="w")
        c.create_text(8, 18, text="* Anillo motor C3", fill=ACCENT2, font=("Courier New",6), anchor="w")
        c.create_text(8, 28, text="| Naka-Rushton C1", fill=YELLOW,  font=("Courier New",6), anchor="w")
        c.create_text(8, 38, text="o C6 Retroceso",    fill=CYAN2,   font=("Courier New",6), anchor="w")
        self._draw_bars()

    def _draw_bars(self):
        bc, nc = self.bar_canvas, self.robot.nc
        bc.delete("all")
        W, bar_w, y = 390, 28, 10
        for name, acts, color, dirs in [
            ("C1 Naka-Rushton", nc.naka_act,  YELLOW,  {d: CARDINALS[d] for d in _DIRS4}),
            ("C2 Gaussianas",   nc.gauss_act,  ACCENT,  {d: CARDINALS[d] for d in _DIRS4}),
            ("C3 Anillo Motor", nc.motor_ring, ACCENT2, {**CARDINALS, **CORNERS}),
        ]:
            bc.create_text(6, y, text=name, fill=color,
                           font=("Courier New", 8, "bold"), anchor="w")
            y += 14
            col_w = W // len(dirs)
            for i, (d, _) in enumerate(dirs.items()):
                act = acts.get(d, 0.0)
                x0 = i*col_w + 4
                h = int(act*34)
                ybot = y + 34
                bc.create_rectangle(x0, y, x0+bar_w, ybot, fill=DARK_GRAY, outline=GRAY)
                if h > 0:
                    bc.create_rectangle(x0, ybot-h, x0+bar_w, ybot, fill=color, outline="")
                bc.create_text(x0+bar_w//2, ybot+7,  text=d,            fill=TEXT,  font=("Courier New",7))
                bc.create_text(x0+bar_w//2, y-7,     text=f"{act:.2f}", fill=color, font=("Courier New",6))
            y += 52

    def _update_ret_bar(self):
        bc, nc = self.ret_bar_canvas, self.robot.nc
        bc.delete("all")
        W = 460
        bc.create_rectangle(0, 0, W, 12, fill=DARK_GRAY, outline=GRAY)
        filled = int(nc.ret_signal * W)
        col = CYAN2 if nc.retroceso_permitido else "#006655"
        if filled > 0:
            bc.create_rectangle(0, 0, filled, 12, fill=col, outline="")
        ux = int(UMBRAL_RET * W)
        bc.create_line(ux, 0, ux, 12, fill=WHITE, width=1, dash=(2,2))

    def start_sim(self):
        if self.running:
            return
        self.running = True
        self.sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self.sim_thread.start()

    def stop_sim(self):
        self.running = False

    def _sim_loop(self):
        MOVE_EVERY = max(1, int(0.5 / DT / 3))
        tick = 0
        while self.running:
            if self.robot.solved:
                self.running = False
                self.root.after(0, lambda: self.decision_detail.config(
                    text="¡Laberinto resuelto!", fg=YELLOW))
                break
            self.robot.sense()
            self.robot.nc.step()
            tick += 1
            if tick >= MOVE_EVERY:
                tick = 0
                self.robot.move()
            self.root.after(0, self._refresh_ui)
            time.sleep(DT)

    def _refresh_ui(self):
        if not self.robot:
            return
        nc = self.robot.nc
        dec = nc.move_decision
        self._draw_robot()
        self._update_neuro_display()
        self._update_ret_bar()

        dec_color = (CYAN2 if nc.retroceso_permitido else
                     MAGENTA if nc.backtracking else GREEN)
        self.decision_label.config(text=dec if dec else "-", fg=dec_color)

        total_cells = self.maze_cols * self.maze_rows
        explored = len(nc.cell_info)
        total_visits = sum(cell.visit_count for cell in nc.cell_info.values())
        repetitions = total_visits - explored

        self.cell_info_label.config(
            text=f"Celdas exploradas: {explored}/{total_cells} | Repeticiones: {repetitions}",
            fg=YELLOW if repetitions < explored * 0.5 else MAGENTA)

        log_disp = str(nc.move_log[-14:]) if nc.move_log else "[]"
        self.mem_label.config(text=f"Movimientos: {log_disp}")
        self.mem_signal_label.config(
            text=f"z[20] mem_signal: {nc.mem_signal:.3f}",
            fg=MAGENTA if nc.mem_signal > 0.5 else YELLOW)

        perm_txt = "SÍ ✓" if nc.retroceso_permitido else "NO"
        perm_col = CYAN2 if nc.retroceso_permitido else GRAY
        self.ret_signal_label.config(
            text=f"z[21] ret_signal: {nc.ret_signal:.3f}  |  Permiso: {perm_txt}",
            fg=perm_col)

        estado = ("BACKTRACK" if nc.backtracking else
                  "C6-RET"   if nc.retroceso_permitido else
                  "BUCLE!"   if nc.loop_detected else
                  "ACTIVO"   if self.running else "DETENIDO")
        self.step_label.config(
            text=f"Pasos: {self.robot.step_count}  |  {estado}  |  "
                 f"osc:{nc.osc_on:.2f}  mem:{nc.mem_signal:.2f}  ret:{nc.ret_signal:.2f}")

        if nc.backtracking:
            self.decision_detail.config(text=f"BACKTRACK → {nc.backtrack_target}")
        elif nc.retroceso_permitido:
            self.decision_detail.config(text=f"C6 ACTIVO — retroceso permitido → {dec}")
        elif dec:
            self.decision_detail.config(text=f"Dirección libre: {dec}")
        else:
            self.decision_detail.config(text="Analizando...")


if __name__ == "__main__":
    root = tk.Tk()
    app = NeuroMazeApp(root)
    root.mainloop()