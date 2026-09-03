import { browser } from '$app/environment';

export type Theme = 'system' | 'light' | 'dark';

const KEY = 'polyrag-theme';

function stored(): Theme {
	if (!browser) return 'system';
	const value = localStorage.getItem(KEY);
	return value === 'light' || value === 'dark' ? value : 'system';
}

export const theme = $state({ value: stored() });

/**
 * Writes the choice to the root element, where the tokens read it.
 *
 * 'system' removes the attribute rather than resolving it to a value, so the
 * page keeps following the OS if the user switches it while the tab is open.
 */
export function setTheme(next: Theme) {
	theme.value = next;
	if (!browser) return;
	if (next === 'system') {
		delete document.documentElement.dataset.theme;
		localStorage.removeItem(KEY);
	} else {
		document.documentElement.dataset.theme = next;
		localStorage.setItem(KEY, next);
	}
}
