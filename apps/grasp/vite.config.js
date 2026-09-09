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

// Hosts whose SPARQL endpoint gets an "Execute on QLever" button, as a
// comma-separated list. An entry is either an exact host ("qlever.dev") or a
// wildcard suffix ("*.qlever.dev", matching any subdomain but not the bare
// domain). Set QLEVER_HOSTS='' to hide the button everywhere.
const QLEVER_HOSTS =
	process.env.QLEVER_HOSTS ??
	'qlever.cs.uni-freiburg.de,qlever.informatik.uni-freiburg.de,qlever.dev,*.qlever.dev';

export default defineConfig({
	plugins: [shareRedirect, sveltekit()],
	define: {
		__API_BASE__: JSON.stringify(process.env.API_BASE || '/api'),
		__QLEVER_HOSTS__: JSON.stringify(
			QLEVER_HOSTS.split(',')
				.map((host) => host.trim())
				.filter(Boolean)
		)
	}
});
