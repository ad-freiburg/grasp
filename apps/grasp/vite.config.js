import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

// Mirror the production nginx /share/:id → /?share=:id redirect in dev, so the
// dev server behaves the same and doesn't 404 on a shared link. (Serving the app
// at /share/:id is not viable anyway — see nginx.conf: Safari ignores <base> for
// dynamic imports. Keeping the app at the mount root works in every browser.)
const shareRedirect = {
	name: 'grasp-share-redirect',
	configureServer(server) {
		server.middlewares.use((req, res, next) => {
			const match = (req.url || '').split('?')[0].match(/^\/share\/([^/]+)\/?$/);
			if (match) {
				res.writeHead(302, { Location: `../?share=${match[1]}` });
				res.end();
				return;
			}
			next();
		});
	}
};

export default defineConfig({
	plugins: [shareRedirect, sveltekit()],
	define: {
		__API_BASE__: JSON.stringify(process.env.API_BASE || '/api')
	}
});
