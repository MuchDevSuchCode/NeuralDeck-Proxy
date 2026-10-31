// NeuralDeck's service worker. It exists so browsers treat the deck as an
// installable app; it caches nothing and leaves every request to the
// network. Only a navigation that fails outright gets a plain offline page.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (e) => {
  if (e.request.mode !== "navigate") return;
  e.respondWith(fetch(e.request).catch(() => new Response(
    "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>" +
    "<title>NeuralDeck</title><body style='background:#05080f;color:#d6e4ff;font:16px system-ui;" +
    "display:flex;align-items:center;justify-content:center;height:100vh;margin:0'>" +
    "<p>NeuralDeck is out of reach. Check the connection and pull to refresh.</p>",
    {status: 503, headers: {"Content-Type": "text/html; charset=utf-8"}})));
});
