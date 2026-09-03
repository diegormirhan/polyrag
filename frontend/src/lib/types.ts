/** UI-side types. API shapes live in api.ts. */

export interface Turn {
	role: 'user' | 'assistant';
	text: string;
	cacheHit?: boolean;
	pending?: boolean;
}
