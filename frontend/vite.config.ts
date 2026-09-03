import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		// Proxied rather than called cross-origin: the browser sees one origin, so
		// there is no CORS config to keep in sync between dev and production.
		proxy: {
			'/api': { target: 'http://127.0.0.1:8000', changeOrigin: true, ws: true },
			'/docs': { target: 'http://127.0.0.1:8000', changeOrigin: true },
			'/openapi.json': { target: 'http://127.0.0.1:8000', changeOrigin: true }
		}
	}
});
