import { api, watchSpans, type Span } from './api';

/**
 * One telemetry connection for the whole app.
 *
 * Every view that shows spans reads from here rather than opening its own socket:
 * a second subscriber would double the traffic and, worse, each view would hold a
 * different slice of history depending on when it mounted.
 */
export const spans = $state<{ items: Span[] }>({ items: [] });

const LIMIT = 1000;
let disconnect: (() => void) | null = null;

export async function connectSpans() {
	if (disconnect) return;

	// Subscribe first, then backfill. The other order drops anything that arrives
	// while the fetch is in flight.
	disconnect = watchSpans((span) => {
		spans.items = [...spans.items.slice(-(LIMIT - 1)), span];
	});

	// The backend keeps a ring buffer of recent spans and GET /telemetry/traces
	// exists to hand it over, but nothing called it: a page load started blank and
	// the Telemetry view looked empty right after a question had been answered.
	try {
		const buffered = await api.traces();
		const live = new Set(spans.items.map((span) => span.span_id));
		const merged = [...buffered.filter((span) => !live.has(span.span_id)), ...spans.items];
		spans.items = merged.slice(-LIMIT);
	} catch {
		// The socket is already connected and is the source that matters; a failed
		// backfill costs history, not function.
	}
}

export function clearSpans() {
	spans.items = [];
}

/** Groups the buffer into traces, newest first. */
export function traces(items: Span[]) {
	const byTrace = new Map<string, Span[]>();
	for (const span of items) {
		const group = byTrace.get(span.trace_id);
		if (group) group.push(span);
		else byTrace.set(span.trace_id, [span]);
	}
	return [...byTrace.entries()]
		.map(([id, group]) => {
			const sorted = [...group].sort((a, b) => a.start_ms - b.start_ms);
			const root = sorted.find((span) => !sorted.some((s) => s.span_id === span.parent_span_id));
			return {
				id,
				spans: sorted,
				start: sorted[0].start_ms,
				name: root?.name ?? sorted[0].name,
				duration: Math.max(...sorted.map((s) => s.start_ms + s.duration_ms)) - sorted[0].start_ms
			};
		})
		.sort((a, b) => b.start - a.start);
}
