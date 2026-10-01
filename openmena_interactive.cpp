#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cstring>

// ============================================================
// OpenMENA Memristor Crossbar Simulator (C++ Backend)
// Communicates with Node.js via stdin/stdout line protocol
// ============================================================

static const int ROWS = 8;
static const int COLS = 8;
static const double G_MIN = 1e-6;   // 1 µS
static const double G_MAX = 1e-3;   // 1 mS
static const double READ_VOLTAGE = 0.1; // 100 mV read pulse

struct CrossbarState {
    double conductance[ROWS][COLS];     // Siemens
    bool   formed[ROWS][COLS];          // Has forming been done?
    int    cycles[ROWS][COLS];          // Endurance cycle count
    double target[ROWS][COLS];          // Target conductance (for VIPI tracking)
    bool   stuck[ROWS][COLS];           // Stuck-at fault

    std::mt19937 rng;
    std::normal_distribution<double> noise;

    CrossbarState() : rng(42), noise(0.0, 0.02) {
        for (int r = 0; r < ROWS; r++) {
            for (int c = 0; c < COLS; c++) {
                conductance[r][c] = G_MIN;
                formed[r][c] = false;
                cycles[r][c] = 0;
                target[r][c] = 0.0;
                stuck[r][c] = false;
            }
        }
    }

    double readWithNoise(int r, int c) {
        double g = conductance[r][c];
        double noisy = g * (1.0 + noise(rng));
        return std::max(G_MIN, std::min(G_MAX, noisy));
    }
};

static CrossbarState xbar;

// Output helpers — Node.js parses lines starting with specific prefixes
void emitCell(int r, int c, double g) {
    // Normalize to 0..1 for the visualizer
    double norm = (g - G_MIN) / (G_MAX - G_MIN);
    norm = std::max(0.0, std::min(1.0, norm));
    std::cout << "CELL " << r << " " << c << " "
              << std::fixed << std::setprecision(9) << g << " "
              << std::fixed << std::setprecision(4) << norm << std::endl;
}

void emitFullGrid() {
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            emitCell(r, c, xbar.conductance[r][c]);
        }
    }
    std::cout << "GRID_DONE" << std::endl;
}

void emitMVMResult(const std::vector<double>& result) {
    std::cout << "MVM";
    for (double v : result) {
        std::cout << " " << std::fixed << std::setprecision(6) << v;
    }
    std::cout << std::endl;
}

void emitVipiStep(int r, int c, int iter, double current_g, double target_g, double error, double voltage) {
    std::cout << "VIPI_STEP " << r << " " << c << " " << iter
              << " " << std::scientific << std::setprecision(4)
              << current_g << " " << target_g << " " << error << " " << voltage
              << std::endl;
}

void emitLog(const std::string& msg) {
    std::cout << "LOG " << msg << std::endl;
}

void emitStats() {
    double total_g = 0, min_g = G_MAX, max_g = G_MIN;
    int formed_count = 0, stuck_count = 0;
    long total_cycles = 0;
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            double g = xbar.conductance[r][c];
            total_g += g;
            if (g < min_g) min_g = g;
            if (g > max_g) max_g = g;
            if (xbar.formed[r][c]) formed_count++;
            if (xbar.stuck[r][c]) stuck_count++;
            total_cycles += xbar.cycles[r][c];
        }
    }
    double avg_g = total_g / (ROWS * COLS);
    std::cout << "STATS "
              << std::scientific << std::setprecision(4)
              << avg_g << " " << min_g << " " << max_g << " "
              << formed_count << " " << stuck_count << " " << total_cycles
              << std::endl;
}

// ============================================================
// Core Operations
// ============================================================

void doForming(int r, int c, double voltage) {
    if (r < 0 || r >= ROWS || c < 0 || c >= COLS) return;
    if (voltage < 2.0) {
        emitLog("Forming voltage too low (need >= 2V)");
        return;
    }
    // Forming creates the conductive filament
    double g_initial = G_MIN * 10.0 + G_MIN * std::abs(xbar.noise(xbar.rng)) * 100.0;
    g_initial = std::max(G_MIN, std::min(G_MAX, g_initial));
    xbar.conductance[r][c] = g_initial;
    xbar.formed[r][c] = true;
    xbar.cycles[r][c]++;
    emitCell(r, c, g_initial);
    emitLog("Formed cell (" + std::to_string(r) + "," + std::to_string(c) + ")");
}

void doSet(int r, int c, double target_g) {
    if (r < 0 || r >= ROWS || c < 0 || c >= COLS) return;
    if (!xbar.formed[r][c]) {
        emitLog("Cell not formed — run FORM first");
        return;
    }
    if (xbar.stuck[r][c]) {
        emitLog("Cell stuck-at fault");
        return;
    }
    target_g = std::max(G_MIN, std::min(G_MAX, target_g));
    xbar.conductance[r][c] = target_g * (1.0 + xbar.noise(xbar.rng) * 0.5);
    xbar.conductance[r][c] = std::max(G_MIN, std::min(G_MAX, xbar.conductance[r][c]));
    xbar.cycles[r][c]++;
    emitCell(r, c, xbar.conductance[r][c]);
}

void doReset(int r, int c) {
    if (r < 0 || r >= ROWS || c < 0 || c >= COLS) return;
    if (xbar.stuck[r][c]) return;
    xbar.conductance[r][c] = G_MIN * (1.0 + std::abs(xbar.noise(xbar.rng)));
    xbar.cycles[r][c]++;
    emitCell(r, c, xbar.conductance[r][c]);
}

// ============================================================
// VIPI Programming — Voltage-Incremental Proportional-Integral
// ============================================================

void doVipi(int r, int c, double target_g, double kp, double ki, double v_inc, int max_iter, double tol) {
    if (r < 0 || r >= ROWS || c < 0 || c >= COLS) return;
    if (!xbar.formed[r][c]) {
        doForming(r, c, 3.0); // Auto-form if needed
    }

    target_g = std::max(G_MIN, std::min(G_MAX, target_g));
    xbar.target[r][c] = target_g;

    double integral_error = 0.0;
    double voltage = 0.5; // Starting voltage

    for (int i = 0; i < max_iter; i++) {
        double current_g = xbar.readWithNoise(r, c);
        double error = target_g - current_g;

        emitVipiStep(r, c, i, current_g, target_g, error, voltage);

        if (std::abs(error / target_g) < tol) {
            emitLog("VIPI converged at iter " + std::to_string(i));
            break;
        }

        // PI control
        integral_error += error;
        double correction = kp * error + ki * integral_error;

        // Voltage-incremental: increase pulse amplitude based on correction
        voltage += v_inc * (error > 0 ? 1.0 : -1.0);
        voltage = std::max(0.1, std::min(3.0, voltage));

        // Apply programming pulse effect
        double delta_g = correction * voltage * 1e-4;
        xbar.conductance[r][c] += delta_g;
        xbar.conductance[r][c] = std::max(G_MIN, std::min(G_MAX, xbar.conductance[r][c]));
        xbar.cycles[r][c]++;
    }

    emitCell(r, c, xbar.conductance[r][c]);
}

// ============================================================
// Matrix-Vector Multiplication: I = G * V
// ============================================================

void doMVM(const std::vector<double>& input_voltages) {
    std::vector<double> output(COLS, 0.0);
    for (int c = 0; c < COLS; c++) {
        double sum = 0.0;
        for (int r = 0; r < ROWS && r < (int)input_voltages.size(); r++) {
            sum += xbar.readWithNoise(r, c) * input_voltages[r];
        }
        output[c] = sum;
    }
    emitMVMResult(output);
}

// ============================================================
// Binary File Operations
// ============================================================

struct BinHeader {
    char magic[4];   // "MENA"
    uint8_t rows;
    uint8_t cols;
    uint8_t version;
    uint8_t flags;
};

void doSaveBin(const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        emitLog("Failed to open file for writing: " + filename);
        return;
    }

    BinHeader hdr;
    std::memcpy(hdr.magic, "MENA", 4);
    hdr.rows = ROWS;
    hdr.cols = COLS;
    hdr.version = 1;
    hdr.flags = 0;
    out.write(reinterpret_cast<char*>(&hdr), sizeof(hdr));

    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            float g = static_cast<float>(xbar.conductance[r][c]);
            out.write(reinterpret_cast<char*>(&g), sizeof(float));
        }
    }
    out.close();
    emitLog("Saved state to " + filename);
}

void doLoadBin(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        emitLog("Failed to open file: " + filename);
        return;
    }

    BinHeader hdr;
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));

    if (std::string(hdr.magic, 4) != "MENA") {
        emitLog("Invalid file format (expected MENA header)");
        return;
    }

    int file_rows = hdr.rows;
    int file_cols = hdr.cols;

    for (int r = 0; r < file_rows && r < ROWS; r++) {
        for (int c = 0; c < file_cols && c < COLS; c++) {
            float g;
            in.read(reinterpret_cast<char*>(&g), sizeof(float));
            xbar.conductance[r][c] = static_cast<double>(g);
            xbar.conductance[r][c] = std::max(G_MIN, std::min(G_MAX, xbar.conductance[r][c]));
            xbar.formed[r][c] = true;
        }
    }
    in.close();
    emitLog("Loaded state from " + filename);
    emitFullGrid();
}

// ============================================================
// Form-all convenience
// ============================================================

void doFormAll(double voltage) {
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            doForming(r, c, voltage);
        }
    }
    emitFullGrid();
}

// ============================================================
// VIPI Program entire matrix from a flat list of target conductances
// ============================================================

void doVipiMatrix(const std::vector<double>& targets) {
    int idx = 0;
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            if (idx < (int)targets.size()) {
                doVipi(r, c, targets[idx], 0.5, 0.1, 0.05, 50, 0.05);
                idx++;
            }
        }
    }
    emitFullGrid();
}

// ============================================================
// Reset entire array
// ============================================================

void doResetAll() {
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            doReset(r, c);
        }
    }
    emitFullGrid();
}

// ============================================================
// Main loop — reads commands from stdin
// ============================================================

int main() {
    emitLog("OpenMENA C++ Backend Started");
    emitLog("Crossbar: " + std::to_string(ROWS) + "x" + std::to_string(COLS));
    emitFullGrid();

    std::string line;
    while (std::getline(std::cin, line)) {
        std::stringstream ss(line);
        std::string cmd;
        ss >> cmd;

        if (cmd == "READ_ALL") {
            emitFullGrid();
        }
        else if (cmd == "STATS") {
            emitStats();
        }
        else if (cmd == "FORM") {
            int r, c;
            double v;
            if (ss >> r >> c >> v) {
                doForming(r, c, v);
            }
        }
        else if (cmd == "FORM_ALL") {
            double v = 3.0;
            ss >> v;
            doFormAll(v);
        }
        else if (cmd == "SET") {
            int r, c;
            double g;
            if (ss >> r >> c >> g) {
                doSet(r, c, g);
            }
        }
        else if (cmd == "RESET") {
            int r, c;
            if (ss >> r >> c) {
                doReset(r, c);
                emitCell(r, c, xbar.conductance[r][c]);
            }
        }
        else if (cmd == "RESET_ALL") {
            doResetAll();
        }
        else if (cmd == "VIPI") {
            int r, c;
            double target_g, kp, ki, v_inc, tol;
            int max_iter;
            if (ss >> r >> c >> target_g >> kp >> ki >> v_inc >> max_iter >> tol) {
                doVipi(r, c, target_g, kp, ki, v_inc, max_iter, tol);
            }
        }
        else if (cmd == "VIPI_MATRIX") {
            // Reads ROWS*COLS float values
            std::vector<double> targets;
            double v;
            while (ss >> v) {
                targets.push_back(v);
            }
            doVipiMatrix(targets);
        }
        else if (cmd == "MVM") {
            std::vector<double> voltages;
            double v;
            while (ss >> v) {
                voltages.push_back(v);
            }
            doMVM(voltages);
        }
        else if (cmd == "SAVE") {
            std::string filename;
            if (ss >> filename) {
                doSaveBin(filename);
            }
        }
        else if (cmd == "LOAD") {
            std::string filename;
            if (ss >> filename) {
                doLoadBin(filename);
            }
        }
        else {
            emitLog("Unknown command: " + cmd);
        }
    }

    return 0;
}
