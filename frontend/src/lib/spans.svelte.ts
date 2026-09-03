import { watchSpans, type Span } from './api';

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

export function connectSpans() {
	if (disconnect) return;
	disconnect = watchSpans((span) => {
		spans.items = [...spans.items.slice(-(LIMIT - 1)), span];
	});
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
