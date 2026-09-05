import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
export default {
	preprocess: vitePreprocess(),
	kit: {
		// A client-side dashboard that talks to FastAPI at runtime — there is no
		// server rendering to do. Static output makes the build a folder of files
		// any server can host, FastAPI itself included.
		adapter: adapter({ fallback: 'index.html' })
	}
};
