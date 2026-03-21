// Service Worker - キャッシュ戦略: Network First (常に最新データを優先)
const CACHE_NAME = 'lunchmap-v1';

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  // POST リクエストやAPI呼び出しはキャッシュしない
  if (event.request.method !== 'GET') return;

  // 静的アセット (CSS, JS, 画像, フォント) のみキャッシュ
  const url = new URL(event.request.url);
  const isStatic = url.pathname.startsWith('/static/') ||
                   url.hostname === 'cdn.jsdelivr.net' ||
                   url.hostname === 'cdn.tailwindcss.com' ||
                   url.hostname === 'unpkg.com';

  if (isStatic) {
    event.respondWith(
      caches.match(event.request).then(cached => {
        const fetchPromise = fetch(event.request).then(response => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
          }
          return response;
        }).catch(() => cached);
        return cached || fetchPromise;
      })
    );
  }
});
