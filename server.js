const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

app.use(express.static('public'));

// ============================================================
// Route: /openmena serves the OpenMENA visualizer
// ============================================================
app.get('/openmena', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'openmena.html'));
});

// ============================================================
// Compile and start the OpenMENA C++ backend
// ============================================================
console.log("Compiling OpenMENA C++ backend...");
const compileMena = spawn('g++', ['-std=c++17', '-pthread', 'openmena_interactive.cpp', '-o', 'openmena_interactive']);

let cppMenaProcess = null;

compileMena.stderr.on('data', (data) => { console.error(`MENA compile stderr: ${data}`); });

compileMena.on('close', (code) => {
    if (code !== 0) {
        console.error(`OpenMENA compilation failed with code ${code}`);
        return;
    }
    console.log("OpenMENA backend compiled successfully.");
    cppMenaProcess = spawn('./openmena_interactive');

    cppMenaProcess.stdout.on('data', (data) => {
        const lines = data.toString().split('\n');
        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;

            if (trimmed.startsWith('CELL ')) {
                // CELL r c g norm
                const parts = trimmed.split(' ');
                io.emit('mena_cell', {
                    r: parseInt(parts[1]),
                    c: parseInt(parts[2]),
                    g: parseFloat(parts[3]),
                    norm: parseFloat(parts[4])
                });
            }
            else if (trimmed === 'GRID_DONE') {
                io.emit('mena_grid_done');
            }
            else if (trimmed.startsWith('MVM')) {
                const parts = trimmed.split(' ').slice(1).map(parseFloat);
                io.emit('mena_mvm', parts);
            }
            else if (trimmed.startsWith('VIPI_STEP ')) {
                const parts = trimmed.split(' ');
                io.emit('mena_vipi_step', {
                    r: parseInt(parts[1]),
                    c: parseInt(parts[2]),
                    iter: parseInt(parts[3]),
                    current_g: parseFloat(parts[4]),
                    target_g: parseFloat(parts[5]),
                    error: parseFloat(parts[6]),
                    voltage: parseFloat(parts[7])
                });
            }
            else if (trimmed.startsWith('LOG ')) {
                io.emit('mena_log', trimmed.substring(4));
            }
            else if (trimmed.startsWith('STATS ')) {
                const parts = trimmed.split(' ');
                io.emit('mena_stats', {
                    avg: parseFloat(parts[1]),
                    min: parseFloat(parts[2]),
                    max: parseFloat(parts[3]),
                    formed: parseInt(parts[4]),
                    stuck: parseInt(parts[5]),
                    cycles: parseInt(parts[6])
                });
            }
        }
    });

    cppMenaProcess.stderr.on('data', (data) => {
        console.error(`OpenMENA C++ Error: ${data}`);
    });
});

// ============================================================
// Socket.IO connections
// ============================================================
io.on('connection', (socket) => {
    console.log('A user connected to the UI');

    // --- OpenMENA Events ---
    socket.on('mena_cmd', (cmd) => {
        if (cppMenaProcess) {
            cppMenaProcess.stdin.write(cmd + '\n');
        }
    });

    // OpenMENA binary file upload — write to temp file and tell C++ to load it
    socket.on('mena_upload_bin', ({ filename, data }) => {
        console.log(`MENA: Received .bin upload: ${filename} (${data.length} bytes)`);
        const tmpPath = path.join(__dirname, '_uploaded_mena.bin');
        fs.writeFileSync(tmpPath, Buffer.from(data));
        if (cppMenaProcess) {
            cppMenaProcess.stdin.write(`LOAD ${tmpPath}\n`);
        }
    });

    // OpenMENA binary download — tell C++ to save, then read and send
    socket.on('mena_download_state', () => {
        const tmpPath = path.join(__dirname, '_download_mena.bin');
        if (cppMenaProcess) {
            cppMenaProcess.stdin.write(`SAVE ${tmpPath}\n`);
            // Give the C++ process a moment to write the file
            setTimeout(() => {
                if (fs.existsSync(tmpPath)) {
                    const data = fs.readFileSync(tmpPath);
                    socket.emit('mena_download', {
                        data: Array.from(data),
                        filename: 'openmena_state.bin'
                    });
                }
            }, 200);
        }
    });
});

// ============================================================
// Start Server
// ============================================================
server.listen(3000, () => {
    console.log('\n=========================================');
    console.log(' Server running at http://localhost:3000');
    console.log(' OpenMENA:       http://localhost:3000/openmena');
    console.log('=========================================\n');
});
