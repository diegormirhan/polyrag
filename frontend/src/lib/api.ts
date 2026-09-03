/** Typed client for the PolyRAG API. Mirrors backend/app/schemas.py. */

export type Route = 'relational' | 'vectorial' | 'graph';

export interface RouteDecision {
	route: Route;
	decision_stage: 'heuristic' | 'embedding' | 'llm_judge';
	margin: number;
	scores: Record<string, number>;
}

export interface Source {
	text?: string;
	score?: number;
	[column: string]: unknown;
}

export interface Span {
	trace_id: string;
	span_id: string;
	parent_span_id: string | null;
	name: string;
	start_ms: number;
	duration_ms: number;
	attributes: Record<string, string | number | boolean>;
}

export interface ServiceHealth {
	name: string;
	url: string;
	up: boolean;
	sleeping: boolean | null;
}

export interface IngestReport {
	file: string;
	chunks: number;
	routes: Route[];
}

/** Events from WS /chat/stream, in the order the pipeline produces them. */
export type ChatEvent =
	| { type: 'cache_hit' }
	| ({ type: 'decision' } & RouteDecision)
	| { type: 'sources'; sources: Source[] }
	| { type: 'token'; text: string }
	| { type: 'done' };

async function json<T>(path: string, init?: RequestInit): Promise<T> {
	const response = await fetch(path, init);
	if (!response.ok) throw new Error(`${init?.method ?? 'GET'} ${path} → ${response.status}`);
	return response.json();
}

export const api = {
	health: () => json<{ ok: boolean; services: ServiceHealth[] }>('/api/v1/health'),
	traces: () => json<Span[]>('/api/v1/telemetry/traces'),
	jobs: () => json<IngestReport[]>('/api/v1/ingest/jobs'),
	clearCache: () => fetch('/api/v1/cache', { method: 'DELETE' }),

	ingest(file: File) {
		const body = new FormData();
		body.append('file', file);
		return json<IngestReport[]>('/api/v1/ingest', { method: 'POST', body });
	}
};

function socket(path: string): WebSocket {
	const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
	return new WebSocket(`${scheme}://${location.host}${path}`);
}

/**
 * Opens the telemetry stream and reconnects if it drops.
 *
 * Returns a stop function. Reconnection matters here because the backend is a
 * local dev server that restarts often — a panel that silently goes dead after
 * the first restart would look like a bug in the panel.
 */
export function watchSpans(onSpan: (span: Span) => void): () => void {
	let ws: WebSocket | null = null;
	let retry: ReturnType<typeof setTimeout> | null = null;
	let stopped = false;

	const connect = () => {
		ws = socket('/api/v1/telemetry/stream');
		ws.onmessage = (event) => onSpan(JSON.parse(event.data));
		ws.onclose = () => {
			if (!stopped) retry = setTimeout(connect, 1500);
		};
	};
	connect();

	return () => {
		stopped = true;
		if (retry) clearTimeout(retry);
		ws?.close();
	};
}

/** Sends one question and yields pipeline events until the answer is complete. */
export async function* streamAnswer(message: string): AsyncGenerator<ChatEvent> {
	const ws = socket('/api/v1/chat/stream');
	const queue: ChatEvent[] = [];
	let notify: (() => void) | null = null;
	let closed = false;

	const wake = () => {
		notify?.();
		notify = null;
	};
	ws.onmessage = (event) => {
		queue.push(JSON.parse(event.data));
		wake();
	};
	ws.onclose = () => {
		closed = true;
		wake();
	};
	ws.onerror = () => {
		closed = true;
		wake();
	};

	await new Promise<void>((resolve, reject) => {
		ws.onopen = () => resolve();
		ws.addEventListener('error', () => reject(new Error('chat stream failed to open')), {
			once: true
		});
	});
	ws.send(JSON.stringify({ message }));

	try {
		while (true) {
			if (queue.length === 0) {
				if (closed) return;
				await new Promise<void>((resolve) => (notify = resolve));
				continue;
			}
			const event = queue.shift()!;
			yield event;
			if (event.type === 'done') return;
		}
	} finally {
		ws.close();
	}
}
