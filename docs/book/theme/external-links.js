// 1. External links open in a new tab so the reader keeps their place.
// 2. Diagram SVGs get a cache-busting query so the browser never shows a
//    stale drawing.
// 3. A self-healing live-reload watchdog: mdBook's own websocket dies on
//    sleep or serve restart and never reconnects; this one reconnects
//    forever and reloads the page when the connection comes back, so
//    changes made while the socket was down are never missed.
document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".content a[href^='http']").forEach(function (link) {
        if (link.hostname !== window.location.hostname) {
            link.target = "_blank";
            link.rel = "noopener noreferrer";
        }
    });

    var stamp = Date.now();
    document.querySelectorAll(".diagram img").forEach(function (img) {
        img.src = img.src.split("?")[0] + "?t=" + stamp;
    });

    if (window.location.protocol === "file:") {
        return; // no server to talk to
    }
    var wasDown = false;
    function connect() {
        var proto = window.location.protocol === "https:" ? "wss://" : "ws://";
        var ws;
        try {
            ws = new WebSocket(proto + window.location.host + "/__livereload");
        } catch (e) {
            wasDown = true;
            setTimeout(connect, 2000);
            return;
        }
        ws.onopen = function () {
            if (wasDown) {
                location.reload(); // catch up on anything missed
            }
        };
        ws.onmessage = function () {
            location.reload();
        };
        ws.onclose = function () {
            wasDown = true;
            setTimeout(connect, 2000);
        };
        ws.onerror = function () {
            ws.close();
        };
    }
    connect();
});
