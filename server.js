/**
 * AutoReel Production Server
 * Node wrapper for instant health check + proxy to Python FastAPI backend.
 * Same pattern as ReelScript — enables CloudPipe Blue-Green zero-downtime deploy.
 */

import { spawn } from 'child_process';
import { createServer } from 'http';
import { join } from 'path';
import httpProxy from 'http-proxy';

const PORT = parseInt(process.env.PORT || '8001', 10);
const BACKEND_PORT = PORT + 1000;
const BACKEND_HOST = `http://127.0.0.1:${BACKEND_PORT}`;

// Create reverse proxy for API/static calls
const proxy = httpProxy.createProxyServer({
	target: BACKEND_HOST,
	ws: true,
	changeOrigin: true,
});

proxy.on('error', (err, _req, res) => {
	console.error('[proxy] error:', err.message);
	if (res.writeHead) {
		res.writeHead(502, { 'Content-Type': 'application/json' });
		res.end(JSON.stringify({ error: 'Backend unavailable' }));
	} else if (res.destroy) {
		res.destroy();
	}
});

// Start Python FastAPI backend via uvicorn
const PYTHON = process.env.PYTHON_PATH || 'C:\\Windows\\py.exe';
const backendEnv = { ...process.env, PORT: String(BACKEND_PORT) };

const backend = spawn(PYTHON, [
	'-m', 'uvicorn', 'main:app',
	'--host', '0.0.0.0',
	'--port', String(BACKEND_PORT),
], {
	cwd: join(import.meta.dirname, 'backend'),
	env: backendEnv,
	stdio: ['ignore', 'pipe', 'pipe'],
	windowsHide: true,
});

backend.stdout.on('data', (data) => {
	process.stdout.write(`[backend] ${data}`);
});

backend.stderr.on('data', (data) => {
	process.stderr.write(`[backend] ${data}`);
});

backend.on('exit', (code) => {
	console.error(`[backend] exited with code ${code}`);
	process.exit(1);
});

// Track backend readiness — Node starts immediately, Python may still be booting
let backendReady = false;

// HTTP server
const server = createServer((req, res) => {
	// Health check responds from Node directly (enables CloudPipe Blue-Green deploy)
	if (req.url === '/api/health') {
		res.writeHead(200, { 'Content-Type': 'application/json' });
		res.end(JSON.stringify({ status: 'ok', service: 'autoreel', backend: backendReady }));
		return;
	}

	// API routes — proxy to Python (503 if not ready)
	if (req.url?.startsWith('/api/')) {
		if (!backendReady) {
			res.writeHead(503, { 'Content-Type': 'application/json' });
			res.end(JSON.stringify({ error: 'Backend starting up, please retry shortly' }));
			return;
		}
		proxy.web(req, res);
		return;
	}

	// Everything else (static HTML, /styles/*, /docs, /) — proxy to Python
	if (!backendReady) {
		res.writeHead(503, { 'Content-Type': 'text/plain' });
		res.end('Service starting up, please retry shortly');
		return;
	}
	proxy.web(req, res);
});

// WebSocket upgrade
server.on('upgrade', (req, socket, head) => {
	if (req.url === '/ws') {
		proxy.ws(req, socket, head);
	} else {
		socket.destroy();
	}
});

// Wait for Python backend to be ready (non-blocking — server already listening)
async function waitForBackend(maxRetries = 30) {
	for (let i = 0; i < maxRetries; i++) {
		try {
			const res = await fetch(`${BACKEND_HOST}/docs`);
			if (res.ok) return true;
		} catch {
			// not ready yet
		}
		await new Promise((r) => setTimeout(r, 1000));
	}
	return false;
}

// Start listening IMMEDIATELY (enables fast Blue-Green health check)
server.listen(PORT, () => {
	console.log(`[server] AutoReel listening on http://localhost:${PORT}`);
	console.log(`[server] Waiting for Python backend on port ${BACKEND_PORT}...`);
});

// Boot Python backend in background
waitForBackend().then((ready) => {
	if (!ready) {
		console.error('[server] Backend failed to start after 30s');
		process.exit(1);
	}
	backendReady = true;
	console.log(`[server] Backend ready — full service active`);
});

// Graceful shutdown
function shutdown() {
	console.log('[server] Shutting down...');
	backend.kill('SIGTERM');
	server.close();
	process.exit(0);
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
