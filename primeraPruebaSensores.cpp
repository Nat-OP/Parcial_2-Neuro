#include <Arduino.h>
#include "Adafruit_VL53L0X.h"
#include <Wire.h>
#include <math.h>
#include <queue>
#include <set>
#include <map>
#include <vector>
#include <deque>

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTES NEURONALES
// ═══════════════════════════════════════════════════════════════════════════════
const float DT = 0.05;
const float TAU_NR = 0.5;
const float TAU_GA = 0.8;
const float TAU_MR = 0.6;
const float TAU_MEM = 2.0;
const float TAU_RET = 1.5;
const float TAU1_OSC = 0.3;
const float TAU2_OSC = 0.8;
const float A_OSC = 2.0;
const float B_OSC = 0.5;
const float C_OSC = 0.6;
const float D_OSC = 0.4;
const float W_EXCIT = 0.6;
const float W_INHIB = 0.3;
const float UMBRAL_RET = 0.65;

// ═══════════════════════════════════════════════════════════════════════════════
// DIRECCIONES Y MAPEOS
// ═══════════════════════════════════════════════════════════════════════════════
const char* _DIRS4[] = {"N", "E", "S", "O"};
const char* _DIRS8[] = {"N", "NE", "E", "SE", "S", "SO", "O", "NO"};

int getDX(const String& dir) {
    if (dir == "N") return 0;
    if (dir == "S") return 0;
    if (dir == "E") return 1;
    if (dir == "O") return -1;
    if (dir == "NE") return 1;
    if (dir == "SE") return 1;
    if (dir == "SO") return -1;
    if (dir == "NO") return -1;
    return 0;
}

int getDY(const String& dir) {
    if (dir == "N") return -1;
    if (dir == "S") return 1;
    if (dir == "E") return 0;
    if (dir == "O") return 0;
    if (dir == "NE") return -1;
    if (dir == "SE") return 1;
    if (dir == "SO") return 1;
    if (dir == "NO") return -1;
    return 0;
}

String getOpposite(const String& dir) {
    if (dir == "N") return "S";
    if (dir == "S") return "N";
    if (dir == "E") return "O";
    if (dir == "O") return "E";
    if (dir == "NE") return "SO";
    if (dir == "SE") return "NO";
    if (dir == "SO") return "NE";
    if (dir == "NO") return "SE";
    return "";
}

float getCardinalAngle(const String& dir) {
    if (dir == "N") return 90;
    if (dir == "E") return 0;
    if (dir == "S") return 270;
    if (dir == "O") return 180;
    if (dir == "NE") return 45;
    if (dir == "SE") return 315;
    if (dir == "SO") return 225;
    if (dir == "NO") return 135;
    return 0;
}

// Índices neuronales
const int I_NR_N = 0;
const int I_NR_E = 1;
const int I_NR_S = 2;
const int I_NR_O = 3;

const int I_GA_N = 4;
const int I_GA_E = 5;
const int I_GA_S = 6;
const int I_GA_O = 7;

const int I_MR_N = 8;
const int I_MR_NE = 9;
const int I_MR_E = 10;
const int I_MR_SE = 11;
const int I_MR_S = 12;
const int I_MR_SO = 13;
const int I_MR_O = 14;
const int I_MR_NO = 15;

const int I_ON = 16;
const int I_OFF = 17;
const int I_ADON = 18;
const int I_ADOF = 19;
const int I_MEM = 20;
const int I_RET = 21;

// ═══════════════════════════════════════════════════════════════════════════════
// PINOUT DE MOTORES Y SENSORES
// ═══════════════════════════════════════════════════════════════════════════════

// Motores
const int STBY1 = 27, STBY2 = 17;
const int MA_PWM = 33, MA_IN1 = 26, MA_IN2 = 25; // Trasera Derecha
const int MB_PWM = 13, MB_IN1 = 14, MB_IN2 = 12; // Delantera Derecha
const int MC_PWM =  0, MC_IN1 = 16, MC_IN2 =  4; // Delantera Izquierda
const int MD_PWM = 32, MD_IN1 =  5, MD_IN2 = 23; // Trasera Izquierda

const int CH_A = 0, CH_B = 1, CH_C = 2, CH_D = 3;
const int VEL_A = 64, VEL_B = 55, VEL_C = 64, VEL_D = 29; // Velocidades base

// Sensores ToF con multiplexor
#define MUX_ADDR 0x70
Adafruit_VL53L0X lox = Adafruit_VL53L0X();

// IMU
TwoWire I2C_IMU = TwoWire(1);
const int SDA_IMU = 18, SCL_IMU = 19, MPU_ADDR = 0x68;

// ═══════════════════════════════════════════════════════════════════════════════
// ESTRUCTURA DE INFORMACIÓN DE CELDA
// ═══════════════════════════════════════════════════════════════════════════════
struct CellInfo {
    int x, y;
    std::set<String> tried;
    int visit_count;
    bool dead_end;
    std::set<String> exits_taken;
    
    CellInfo(int _x, int _y) : x(_x), y(_y), visit_count(0), dead_end(false) {}
    
    // Para usar como clave en map
    bool operator<(const CellInfo& other) const {
        if (x != other.x) return x < other.x;
        return y < other.y;
    }
};

// Comparador para pair<int,int>
struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const {
        return p.first * 1000 + p.second;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// CONTROLADOR NEURONAL CON MEMORIA ANTI-REPETICIÓN
// ═══════════════════════════════════════════════════════════════════════════════
class NeuroController {
private:
    // Variables de estado
    float z[22];
    std::map<String, float> s_prox;
    std::vector<String> open_dirs;
    int current_x, current_y;
    
    // Capas de activación
    std::map<String, float> naka_act;
    std::map<String, float> gauss_act;
    std::map<String, float> motor_ring;
    
    // Memoria de exploración
    std::map<std::pair<int, int>, CellInfo> cell_info;
    std::map<std::pair<int, int>, int> visit_count;
    std::vector<String> move_log;
    std::deque<String> recent_moves;
    String last_move;
    String move_decision;
    
    // Sistema de backtracking mejorado
    bool backtracking;
    std::pair<int, int> backtrack_target;
    std::vector<String> backtrack_path;
    bool checkpoint_active;
    bool retroceso_permitido;
    
    // Detección de ciclos
    bool loop_detected;
    int no_progress_counter;
    std::pair<int, int> last_position;
    
    // Propiedades UI
    float osc_on;
    float osc_off;
    float mem_signal;
    float ret_signal;
    
    // Métodos privados
    bool _needs_backtrack();
    std::tuple<int, int, std::vector<String>> _find_unexplored_path();
    void _start_backtrack();
    void _reset_exploration();
    void _execute_backtrack();
    void _calculate_motor_ring();
    String _choose_direction();
    
public:
    NeuroController();
    void update_sensors(std::map<String, float> distances);
    void set_open_dirs(std::vector<String> dirs, int x, int y);
    void step(float dt = DT);
    void record_move(String dir);
    
    // Getters
    float get_osc_on() { return osc_on; }
    float get_osc_off() { return osc_off; }
    float get_mem_signal() { return mem_signal; }
    float get_ret_signal() { return ret_signal; }
    String get_move_decision() { return move_decision; }
    std::map<String, float> get_naka_act() { return naka_act; }
    std::map<String, float> get_gauss_act() { return gauss_act; }
    std::map<String, float> get_motor_ring() { return motor_ring; }
};

// Implementación del NeuroController
NeuroController::NeuroController() {
    for (int i = 0; i < 22; i++) z[i] = 0.0;
    
    s_prox["N"] = 0.0; s_prox["E"] = 0.0; s_prox["S"] = 0.0; s_prox["O"] = 0.0;
    naka_act["N"] = 0.0; naka_act["E"] = 0.0; naka_act["S"] = 0.0; naka_act["O"] = 0.0;
    gauss_act["N"] = 0.0; gauss_act["E"] = 0.0; gauss_act["S"] = 0.0; gauss_act["O"] = 0.0;
    
    for (int i = 0; i < 8; i++) {
        motor_ring[_DIRS8[i]] = 0.0;
    }
    
    current_x = 0; current_y = 0;
    last_move = "";
    move_decision = "";
    backtracking = false;
    backtrack_target = std::make_pair(0, 0);
    checkpoint_active = false;
    retroceso_permitido = false;
    loop_detected = false;
    no_progress_counter = 0;
    last_position = std::make_pair(0, 0);
    osc_on = 0.0; osc_off = 0.0; mem_signal = 0.0; ret_signal = 0.0;
}

void NeuroController::update_sensors(std::map<String, float> distances) {
    float max_dist = 400.0;
    for (int i = 0; i < 4; i++) {
        String d = _DIRS4[i];
        s_prox[d] = min(1.0, distances[d] / max_dist);
    }
}

void NeuroController::set_open_dirs(std::vector<String> dirs, int x, int y) {
    open_dirs = dirs;
    current_x = x;
    current_y = y;
    
    auto pos = std::make_pair(x, y);
    
    if (cell_info.find(pos) == cell_info.end()) {
        cell_info[pos] = CellInfo(x, y);
    }
    
    cell_info[pos].visit_count++;
    visit_count[pos] = cell_info[pos].visit_count;
    
    if (dirs.size() == 1 && cell_info[pos].visit_count > 1) {
        cell_info[pos].dead_end = true;
    }
}

void NeuroController::record_move(String dir) {
    if (dir != "") {
        last_move = dir;
        move_log.push_back(dir);
        recent_moves.push_back(dir);
        
        // Actualizar celda actual con dirección tomada
        auto pos = std::make_pair(current_x, current_y);
        if (cell_info.find(pos) != cell_info.end()) {
            cell_info[pos].tried.insert(dir);
        }
        
        // Actualizar posición
        current_x += getDX(dir);
        current_y += getDY(dir);
    }
}

bool NeuroController::_needs_backtrack() {
    if (backtracking) return true;
    
    if (open_dirs.size() == 0) return true;
    
    auto pos = std::make_pair(current_x, current_y);
    int current_visits = (visit_count.find(pos) != visit_count.end()) ? visit_count[pos] : 0;
    
    if (current_visits > 3) {
        auto it = cell_info.find(pos);
        if (it != cell_info.end()) {
            int unexplored = 0;
            for (auto d : open_dirs) {
                if (it->second.tried.find(d) == it->second.tried.end()) unexplored++;
            }
            if (unexplored == 0) return true;
        }
    }
    
    if (recent_moves.size() >= 8) {
        String last_moves = "";
        auto it = recent_moves.begin();
        std::advance(it, recent_moves.size() - 6);
        for (int i = 0; i < 6 && it != recent_moves.end(); i++) {
            last_moves += *it;
            it++;
        }
        
        if (last_moves.indexOf("NSNS") >= 0 || last_moves.indexOf("SNSN") >= 0 ||
            last_moves.indexOf("EOEO") >= 0 || last_moves.indexOf("OEOE") >= 0) {
            loop_detected = true;
            return true;
        }
    }
    
    return false;
}

std::tuple<int, int, std::vector<String>> NeuroController::_find_unexplored_path() {
    std::queue<std::tuple<int, int, std::vector<String>>> queue;
    std::set<std::pair<int, int>> visited;
    
    queue.push(std::make_tuple(current_x, current_y, std::vector<String>()));
    visited.insert(std::make_pair(current_x, current_y));
    
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop();
        
        int x = std::get<0>(current);
        int y = std::get<1>(current);
        std::vector<String> path = std::get<2>(current);
        
        auto pos = std::make_pair(x, y);
        auto it = cell_info.find(pos);
        
        if (it != cell_info.end()) {
            for (auto d : _DIRS4) {
                // Verificar si es dirección válida (simplificado)
                if (it->second.tried.find(d) == it->second.tried.end()) {
                    if (!it->second.dead_end) {
                        return std::make_tuple(x, y, path);
                    }
                }
            }
        }
        
        for (int i = 0; i < 4; i++) {
            String d = _DIRS4[i];
            int nx = x + getDX(d);
            int ny = y + getDY(d);
            auto new_pos = std::make_pair(nx, ny);
            
            if (visited.find(new_pos) == visited.end()) {
                if (cell_info.find(new_pos) != cell_info.end() || (nx >= -50 && nx <= 50 && ny >= -50 && ny <= 50)) {
                    visited.insert(new_pos);
                    std::vector<String> new_path = path;
                    new_path.push_back(d);
                    queue.push(std::make_tuple(nx, ny, new_path));
                }
            }
        }
    }
    
    return std::make_tuple(0, 0, std::vector<String>());
}

void NeuroController::_start_backtrack() {
    if (backtracking) return;
    
    auto result = _find_unexplored_path();
    int target_x = std::get<0>(result);
    int target_y = std::get<1>(result);
    std::vector<String> path = std::get<2>(result);
    
    if (target_x != 0 || target_y != 0 || path.size() > 0) {
        backtracking = true;
        backtrack_target = std::make_pair(target_x, target_y);
        backtrack_path = path;
        Serial.printf("Backtrack iniciado hacia (%d,%d)\n", target_x, target_y);
    } else {
        _reset_exploration();
    }
}

void NeuroController::_reset_exploration() {
    for (auto& entry : cell_info) {
        if (entry.second.visit_count > 2 && !entry.second.dead_end) {
            entry.second.tried.clear();
        }
    }
    backtracking = false;
    loop_detected = false;
    no_progress_counter = 0;
    Serial.println("Reset parcial de exploración");
}

void NeuroController::_execute_backtrack() {
    if (backtrack_target.first == 0 && backtrack_target.second == 0) {
        backtracking = false;
        return;
    }
    
    int tx = backtrack_target.first;
    int ty = backtrack_target.second;
    int dx = tx - current_x;
    int dy = ty - current_y;
    
    String target_dir;
    if (abs(dx) > abs(dy)) {
        target_dir = (dx > 0) ? "E" : "O";
    } else {
        target_dir = (dy > 0) ? "S" : "N";
    }
    
    bool dir_valid = false;
    for (auto d : open_dirs) {
        if (d == target_dir) {
            dir_valid = true;
            break;
        }
    }
    
    if (dir_valid) {
        move_decision = target_dir;
        return;
    } else {
        for (auto d : open_dirs) {
            if (d != getOpposite(last_move)) {
                move_decision = d;
                return;
            }
        }
    }
    
    if (current_x == tx && current_y == ty) {
        backtracking = false;
        backtrack_target = std::make_pair(0, 0);
        backtrack_path.clear();
    }
}

void NeuroController::_calculate_motor_ring() {
    std::map<String, float> motor_vals;
    
    for (int i = 0; i < 4; i++) {
        String d = _DIRS4[i];
        float base_act = gauss_act[d];
        auto pos = std::make_pair(current_x, current_y);
        auto it = cell_info.find(pos);
        
        if (it != cell_info.end() && it->second.tried.find(d) != it->second.tried.end()) {
            base_act *= 0.3;
        }
        
        int nx = current_x + getDX(d);
        int ny = current_y + getDY(d);
        auto neighbor_pos = std::make_pair(nx, ny);
        auto neighbor_it = cell_info.find(neighbor_pos);
        
        if (neighbor_it != cell_info.end() && neighbor_it->second.dead_end) {
            base_act *= 0.1;
        }
        
        if (neighbor_it != cell_info.end() && neighbor_it->second.visit_count > 2) {
            base_act *= max(0.2f, 1.0f - neighbor_it->second.visit_count * 0.2f);
        }
        
        motor_vals[d] = base_act;
    }
    
    // Calcular esquinas
    struct Corner { String name; String d1; String d2; };
    Corner corners[] = {
        {"NE", "N", "E"}, {"SE", "S", "E"},
        {"SO", "S", "O"}, {"NO", "N", "O"}
    };
    
    for (auto& corner : corners) {
        motor_vals[corner.name] = sqrt(motor_vals[corner.d1] * motor_vals[corner.d2]);
    }
    
    for (int i = 0; i < 8; i++) {
        String d = _DIRS8[i];
        motor_ring[d] = (motor_vals.find(d) != motor_vals.end()) ? motor_vals[d] : 0.0;
    }
}

String NeuroController::_choose_direction() {
    if (move_log.size() == 0) {
        if (open_dirs.size() > 0) {
            String best_dir = open_dirs[0];
            float max_act = motor_ring[best_dir];
            for (auto d : open_dirs) {
                if (motor_ring[d] > max_act) {
                    max_act = motor_ring[d];
                    best_dir = d;
                }
            }
            return best_dir;
        }
        return "";
    }
    
    auto pos = std::make_pair(current_x, current_y);
    std::vector<String> unexplored;
    
    auto it = cell_info.find(pos);
    if (it != cell_info.end()) {
        for (auto d : open_dirs) {
            if (it->second.tried.find(d) == it->second.tried.end()) {
                unexplored.push_back(d);
            }
        }
    }
    
    String back_dir = (last_move != "") ? getOpposite(last_move) : "";
    
    if (unexplored.size() > 0) {
        std::vector<String> valid_dirs;
        for (auto d : unexplored) {
            if (d != back_dir) valid_dirs.push_back(d);
        }
        
        if (valid_dirs.size() > 0) {
            String best_dir = valid_dirs[0];
            float max_act = motor_ring[best_dir];
            for (auto d : valid_dirs) {
                if (motor_ring[d] > max_act) {
                    max_act = motor_ring[d];
                    best_dir = d;
                }
            }
            return best_dir;
        } else {
            String best_dir = unexplored[0];
            float max_act = motor_ring[best_dir];
            for (auto d : unexplored) {
                if (motor_ring[d] > max_act) {
                    max_act = motor_ring[d];
                    best_dir = d;
                }
            }
            return best_dir;
        }
    } else if (open_dirs.size() > 0) {
        std::vector<String> valid_dirs;
        for (auto d : open_dirs) {
            if (d != back_dir) valid_dirs.push_back(d);
        }
        
        if (valid_dirs.size() > 0) {
            String best_dir = valid_dirs[0];
            float max_act = motor_ring[best_dir];
            for (auto d : valid_dirs) {
                if (motor_ring[d] > max_act) {
                    max_act = motor_ring[d];
                    best_dir = d;
                }
            }
            return best_dir;
        } else {
            String best_dir = open_dirs[0];
            float max_act = motor_ring[best_dir];
            for (auto d : open_dirs) {
                if (motor_ring[d] > max_act) {
                    max_act = motor_ring[d];
                    best_dir = d;
                }
            }
            return best_dir;
        }
    }
    
    return "";
}

// Función de activación Naka-Rushton
float naka_rushton_f(float x, float M, float n, float C) {
    if (x <= 0) return 0.0;
    return M * pow(x, n) / (pow(x, n) + pow(C, n));
}

// Función Gaussiana
float gaussian_f(float theta1, float theta2, float sigma) {
    float diff = fabs(theta1 - theta2);
    if (diff > 180) diff = 360 - diff;
    return exp(-pow(diff, 2) / (2 * pow(sigma, 2)));
}

void NeuroController::step(float dt) {
    // C1 — Naka-Rushton
    String dirs4[] = {"N", "E", "S", "O"};
    int nr_indices[] = {I_NR_N, I_NR_E, I_NR_S, I_NR_O};
    
    for (int i = 0; i < 4; i++) {
        float f = naka_rushton_f(s_prox[dirs4[i]], 1.0, 2.0, 0.2);
        z[nr_indices[i]] = z[nr_indices[i]] + (dt/TAU_NR) * (-z[nr_indices[i]] + f);
        z[nr_indices[i]] = max(0.0f, min(1.0f, z[nr_indices[i]]));
        naka_act[dirs4[i]] = z[nr_indices[i]];
    }
    
    // C2 — Gaussianas con convolución inhibitoria
    float nr[4];
    for (int i = 0; i < 4; i++) {
        nr[i] = z[nr_indices[i]];
    }
    
    int ga_indices[] = {I_GA_N, I_GA_E, I_GA_S, I_GA_O};
    
    for (int i = 0; i < 4; i++) {
        int im1 = (i - 1 + 4) % 4;
        int ip1 = (i + 1) % 4;
        int ip2 = (i + 2) % 4;
        
        float u = (gaussian_f(getCardinalAngle(dirs4[i]), getCardinalAngle(dirs4[i]), 45.0) * nr[i]
                 + W_EXCIT * gaussian_f(getCardinalAngle(dirs4[i]), getCardinalAngle(dirs4[im1]), 45.0) * nr[im1]
                 + W_EXCIT * gaussian_f(getCardinalAngle(dirs4[i]), getCardinalAngle(dirs4[ip1]), 45.0) * nr[ip1]
                 - W_INHIB * gaussian_f(getCardinalAngle(dirs4[i]), getCardinalAngle(dirs4[ip2]), 45.0) * nr[ip2]);
        
        u = max(0.0f, min(1.0f, u));
        z[ga_indices[i]] = z[ga_indices[i]] + (dt/TAU_GA) * (-z[ga_indices[i]] + u);
        z[ga_indices[i]] = max(0.0f, min(1.0f, z[ga_indices[i]]));
        gauss_act[dirs4[i]] = z[ga_indices[i]];
    }
    
    // C3 — Anillo motor con penalización
    _calculate_motor_ring();
    
    int mr_indices[] = {I_MR_N, I_MR_NE, I_MR_E, I_MR_SE, I_MR_S, I_MR_SO, I_MR_O, I_MR_NO};
    for (int i = 0; i < 8; i++) {
        z[mr_indices[i]] = z[mr_indices[i]] + (dt/TAU_MR) * (-z[mr_indices[i]] + motor_ring[_DIRS8[i]]);
        z[mr_indices[i]] = max(0.0f, min(1.0f, z[mr_indices[i]]));
    }
    
    // C4 — Oscilador Wilson-Cowan
    float K1 = z[I_MR_NE] + z[I_MR_NO];
    float K2 = z[I_MR_SE] + z[I_MR_SO];
    float zn = (K1 + K2) / 2.0f + 1e-6f;
    
    float u_on = max(0.0f, zn * K1 - D_OSC * z[I_OFF]);
    float u_off = max(0.0f, zn * K2 - D_OSC * z[I_ON]);
    
    z[I_ON] += (dt/TAU1_OSC) * (-z[I_ON] + A_OSC * pow(u_on, 2) / (pow(B_OSC + z[I_ADON], 2) + pow(u_on, 2) + 1e-12f));
    z[I_OFF] += (dt/TAU1_OSC) * (-z[I_OFF] + A_OSC * pow(u_off, 2) / (pow(B_OSC + z[I_ADOF], 2) + pow(u_off, 2) + 1e-12f));
    z[I_ADON] += (dt/TAU2_OSC) * (-z[I_ADON] + C_OSC * z[I_ON]);
    z[I_ADOF] += (dt/TAU2_OSC) * (-z[I_ADOF] + C_OSC * z[I_OFF]);
    
    int osc_indices[] = {I_ON, I_OFF, I_ADON, I_ADOF};
    for (int i = 0; i < 4; i++) {
        z[osc_indices[i]] = max(0.0f, z[osc_indices[i]]);
    }
    
    osc_on = z[I_ON];
    osc_off = z[I_OFF];
    
    // C5 — Señal de callejón y detección de bucles
    String back_dir = (last_move != "") ? getOpposite(last_move) : "";
    std::vector<String> fwd;
    for (auto d : open_dirs) {
        if (d != back_dir) fwd.push_back(d);
    }
    
    float loop_penalty = loop_detected ? 1.0f : 0.0f;
    float dead = (fwd.size() == 0 && !backtracking) ? 1.0f : 0.0f;
    float mem_input = max(dead, loop_penalty);
    z[I_MEM] = z[I_MEM] + (dt/TAU_MEM) * (-z[I_MEM] + mem_input);
    z[I_MEM] = max(0.0f, min(1.0f, z[I_MEM]));
    mem_signal = z[I_MEM];
    
    // C6 — Permiso de retroceso
    float ret = _needs_backtrack() ? 1.0f : 0.0f;
    z[I_RET] = z[I_RET] + (dt/TAU_RET) * (-z[I_RET] + ret);
    z[I_RET] = max(0.0f, min(1.0f, z[I_RET]));
    ret_signal = z[I_RET];
    retroceso_permitido = (z[I_RET] > UMBRAL_RET);
    
    // Tomar decisión de movimiento
    if (backtracking) {
        _execute_backtrack();
    } else if (retroceso_permitido && open_dirs.size() == 0) {
        _start_backtrack();
    } else {
        if (move_log.size() == 0) {
            backtracking = false;
            retroceso_permitido = false;
        }
        move_decision = _choose_direction();
    }
    
    // Actualizar detección de progreso
    auto current_pos = std::make_pair(current_x, current_y);
    if (current_pos == last_position) {
        no_progress_counter++;
    } else {
        no_progress_counter = 0;
        last_position = current_pos;
    }
    
    if (no_progress_counter > 20) {
        loop_detected = true;
        _start_backtrack();
        no_progress_counter = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FUNCIONES DE MOTORES
// ═══════════════════════════════════════════════════════════════════════════════
void controlMotor(int canal, int pin1, int pin2, int vel) {
    digitalWrite(pin1, vel > 0 ? HIGH : LOW);
    digitalWrite(pin2, vel < 0 ? HIGH : LOW);
    ledcWrite(canal, abs(vel));
}

void traseraDerecha(int v)     { controlMotor(CH_A, MA_IN1, MA_IN2, -v); }
void delanteraDerecha(int v)   { controlMotor(CH_B, MB_IN1, MB_IN2,  v); }
void delanteraIzquierda(int v) { controlMotor(CH_C, MC_IN1, MC_IN2, -v); }
void traseraIzquierda(int v)   { controlMotor(CH_D, MD_IN1, MD_IN2, -v); }

void frenarTodos() {
    traseraDerecha(0); delanteraDerecha(0);
    delanteraIzquierda(0); traseraIzquierda(0);
}

void avanzar() {
    traseraDerecha(VEL_A); delanteraDerecha(VEL_B);
    delanteraIzquierda(VEL_C); traseraIzquierda(VEL_D);
}

void girarDerecha() {
    traseraDerecha(-65); delanteraDerecha(-65);
    delanteraIzquierda(65); traseraIzquierda(65);
}

void girarIzquierda() {
    traseraDerecha(65); delanteraDerecha(65);
    delanteraIzquierda(-65); traseraIzquierda(-65);
}

void retroceder() {
    traseraDerecha(-VEL_A); delanteraDerecha(-VEL_B);
    delanteraIzquierda(-VEL_C); traseraIzquierda(-VEL_D);
}

// ═══════════════════════════════════════════════════════════════════════════════
// FUNCIONES PARA SENSORES ToF CON MULTIPLEXOR
// ═══════════════════════════════════════════════════════════════════════════════
void tcaselect(uint8_t i) {
    if (i > 7) return;
    Wire.beginTransmission(MUX_ADDR);
    Wire.write(1 << i);
    Wire.endTransmission();
}

// Mapeo de canales a direcciones: 0=Norte, 1=Este, 2=Sur, 3=Oeste
float leerSensorToF(uint8_t canal) {
    VL53L0X_RangingMeasurementData_t measure;
    tcaselect(canal);
    lox.rangingTest(&measure, false);
    
    if (measure.RangeStatus != 4) {
        return measure.RangeMilliMeter;
    }
    return 400.0; // Retornar distancia máxima si no hay lectura
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURACIÓN DE LEDC PARA PWM DE MOTORES
// ═══════════════════════════════════════════════════════════════════════════════
void setupMotores() {
    ledcSetup(CH_A, 5000, 8);
    ledcSetup(CH_B, 5000, 8);
    ledcSetup(CH_C, 5000, 8);
    ledcSetup(CH_D, 5000, 8);
    
    ledcAttachPin(MA_PWM, CH_A);
    ledcAttachPin(MB_PWM, CH_B);
    ledcAttachPin(MC_PWM, CH_C);
    ledcAttachPin(MD_PWM, CH_D);
    
    pinMode(MA_IN1, OUTPUT);
    pinMode(MA_IN2, OUTPUT);
    pinMode(MB_IN1, OUTPUT);
    pinMode(MB_IN2, OUTPUT);
    pinMode(MC_IN1, OUTPUT);
    pinMode(MC_IN2, OUTPUT);
    pinMode(MD_IN1, OUTPUT);
    pinMode(MD_IN2, OUTPUT);
    pinMode(STBY1, OUTPUT);
    pinMode(STBY2, OUTPUT);
    
    digitalWrite(STBY1, HIGH);
    digitalWrite(STBY2, HIGH);
}

// ═══════════════════════════════════════════════════════════════════════════════
// VARIABLES GLOBALES
// ═══════════════════════════════════════════════════════════════════════════════
NeuroController neuro;
int pos_x = 0, pos_y = 0;
unsigned long lastStep = 0;

// ═══════════════════════════════════════════════════════════════════════════════
// SETUP
// ═══════════════════════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    Wire.begin();
    
    // Inicializar motores
    setupMotores();
    
    // Inicializar sensores ToF con multiplexor
    Serial.println("Iniciando Sensores ToF con Multiplexor...");
    for (uint8_t i = 0; i < 4; i++) {
        tcaselect(i);
        if (!lox.begin()) {
            Serial.printf("Fallo en canal %d\n", i);
        } else {
            Serial.printf("Sensor %d OK\n", i);
        }
    }
    
    Serial.println("Sistema listo");
    lastStep = millis();
}

// ═══════════════════════════════════════════════════════════════════════════════
// FUNCIÓN PARA DETERMINAR DIRECCIONES ABIERTAS BASADO EN SENSORES
// ═══════════════════════════════════════════════════════════════════════════════
std::vector<String> determinarDireccionesAbiertas() {
    std::vector<String> abiertas;
    float umbral_pared = 150.0; // mm - si hay obstáculo a menos de 150mm, consideramos pared
    
    // Leer sensores: canal 0=Norte, 1=Este, 2=Sur, 3=Oeste
    float distN = leerSensorToF(0);
    float distE = leerSensorToF(1);
    float distS = leerSensorToF(2);
    float distO = leerSensorToF(3);
    
    // Para debug
    Serial.printf("Sensores: N=%.0f E=%.0f S=%.0f O=%.0f mm\n", distN, distE, distS, distO);
    
    if (distN > umbral_pared) abiertas.push_back("N");
    if (distE > umbral_pared) abiertas.push_back("E");
    if (distS > umbral_pared) abiertas.push_back("S");
    if (distO > umbral_pared) abiertas.push_back("O");
    
    return abiertas;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FUNCIÓN PARA EJECUTAR MOVIMIENTO
// ═══════════════════════════════════════════════════════════════════════════════
void ejecutarMovimiento(String direccion, float duracion_ms = 500) {
    frenarTodos();
    delay(50);
    
    if (direccion == "N") {
        avanzar();
    } else if (direccion == "S") {
        retroceder();
    } else if (direccion == "E") {
        girarDerecha();
    } else if (direccion == "O") {
        girarIzquierda();
    } else if (direccion == "NE") {
        // Diagonal noreste: combinar N y E
        traseraDerecha(VEL_A); delanteraDerecha(VEL_B);
        delanteraIzquierda(VEL_C); traseraIzquierda(VEL_D);
    } else if (direccion == "NO") {
        // Diagonal noroeste: combinar N y O
        traseraDerecha(VEL_A); delanteraDerecha(VEL_B);
        delanteraIzquierda(VEL_C); traseraIzquierda(VEL_D);
    } else if (direccion == "SE") {
        // Diagonal sureste: combinar S y E
        traseraDerecha(-VEL_A); delanteraDerecha(-VEL_B);
        delanteraIzquierda(-VEL_C); traseraIzquierda(-VEL_D);
    } else if (direccion == "SO") {
        // Diagonal suroeste: combinar S y O
        traseraDerecha(-VEL_A); delanteraDerecha(-VEL_B);
        delanteraIzquierda(-VEL_C); traseraIzquierda(-VEL_D);
    }
    
    delay(duracion_ms);
    frenarTodos();
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOOP PRINCIPAL
// ═══════════════════════════════════════════════════════════════════════════════
void loop() {
    unsigned long ahora = millis();
    
    // Actualizar sensores y red neuronal cada 50ms
    if (ahora - lastStep >= 50) {
        lastStep = ahora;
        
        // 1. Leer sensores ToF
        std::map<String, float> distancias;
        distancias["N"] = leerSensorToF(0);
        distancias["E"] = leerSensorToF(1);
        distancias["S"] = leerSensorToF(2);
        distancias["O"] = leerSensorToF(3);
        
        // 2. Actualizar sensores en la red neuronal
        neuro.update_sensors(distancias);
        
        // 3. Determinar direcciones abiertas
        std::vector<String> abiertas = determinarDireccionesAbiertas();
        
        // 4. Actualizar estado de posición y direcciones abiertas
        neuro.set_open_dirs(abiertas, pos_x, pos_y);
        
        // 5. Ejecutar paso de la red neuronal
        neuro.step(DT);
        
        // 6. Obtener decisión de movimiento
        String decision = neuro.get_move_decision();
        
        // 7. Ejecutar movimiento si hay decisión válida
        if (decision != "") {
            Serial.printf("Movimiento: %s\n", decision.c_str());
            ejecutarMovimiento(decision, 400);
            
            // Registrar movimiento y actualizar posición
            neuro.record_move(decision);
            
            // Actualizar coordenadas según movimiento
            if (decision == "N") pos_y--;
            else if (decision == "S") pos_y++;
            else if (decision == "E") pos_x++;
            else if (decision == "O") pos_x--;
            else if (decision == "NE") { pos_x++; pos_y--; }
            else if (decision == "SE") { pos_x++; pos_y++; }
            else if (decision == "SO") { pos_x--; pos_y++; }
            else if (decision == "NO") { pos_x--; pos_y--; }
            
            Serial.printf("Posición actual: (%d, %d)\n", pos_x, pos_y);
        }
        
        // 8. Mostrar estado neuronal para debug
        Serial.printf("Retroceso: %.2f | Mem: %.2f | OSC: %.2f\n", 
                      neuro.get_ret_signal(), neuro.get_mem_signal(), neuro.get_osc_on());
    }
    
    // Pequeña pausa para no saturar
    delay(10);
}