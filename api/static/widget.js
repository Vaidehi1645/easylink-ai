/* EasyLink AI embeddable widget (Phase 2)
 * One line to embed: <script src=".../widget.js" data-site="TOKEN"></script>
 * Shadow DOM keeps the host page's CSS out and ours in (hostile-CSS proof).
 */
(function () {
  "use strict";

  var script =
    document.currentScript ||
    Array.prototype.find.call(document.scripts, function (s) {
      return s.src && s.hasAttribute("data-site");
    });
  if (!script) {
    console.warn("[EasyLink] missing data-site on the script tag");
    return;
  }
  var token = script.getAttribute("data-site");
  var base = new URL(script.src, location.href).origin;
  var history = [];

  function mount() {
    var host = document.createElement("div");
    document.body.appendChild(host);
    var shadow = host.attachShadow({ mode: "open" });

    shadow.innerHTML =
      '<style>' +
      "*{box-sizing:border-box;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;letter-spacing:normal;text-transform:none;color:#111827}" +
      ".bubble{position:fixed;right:20px;bottom:20px;width:56px;height:56px;border:none;border-radius:50%;background:#4f46e5;color:#fff;font-size:24px;cursor:pointer;box-shadow:0 6px 20px rgba(0,0,0,.25);z-index:2147483000;transition:transform .15s}" +
      ".bubble:hover{transform:scale(1.06)}" +
      ".panel{position:fixed;right:20px;bottom:88px;width:340px;max-width:calc(100vw - 40px);height:460px;max-height:calc(100vh - 130px);background:#fff;border-radius:14px;box-shadow:0 12px 40px rgba(0,0,0,.28);display:flex;flex-direction:column;overflow:hidden;z-index:2147483000;border:1px solid #e5e7eb}" +
      ".panel[hidden]{display:none}" +
      ".hdr{background:#4f46e5;color:#fff;padding:11px 13px;font-weight:600;font-size:14.5px;display:flex;align-items:center;justify-content:space-between;gap:8px}" +
      ".hdr .brand{font-weight:400;font-size:10.5px;opacity:.9}" +
      ".hdr .brand a{color:#fff}" +
      ".hdr .x{cursor:pointer;background:none;border:none;color:#fff;font-size:17px;line-height:1;padding:2px}" +
      ".msgs{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px;background:#f9fafb}" +
      ".m{max-width:88%;padding:8px 11px;border-radius:12px;font-size:13.5px;line-height:1.45;white-space:pre-wrap;word-break:break-word}" +
      ".u{align-self:flex-end;background:#4f46e5;color:#fff;border-bottom-right-radius:4px}" +
      ".b{align-self:flex-start;background:#fff;color:#111827;border:1px solid #e5e7eb;border-bottom-left-radius:4px}" +
      ".src{display:block;margin-top:6px;font-size:11px}" +
      ".src a{color:#4f46e5;text-decoration:none;display:block;margin-top:2px}" +
      ".src a:hover{text-decoration:underline}" +
      ".err{color:#b91c1c;font-size:12.5px}" +
      ".typing span{display:inline-block;width:6px;height:6px;margin-right:3px;background:#9ca3af;border-radius:50%;animation:elb 1.2s infinite}" +
      ".typing span:nth-child(2){animation-delay:.2s}" +
      ".typing span:nth-child(3){animation-delay:.4s}" +
      "@keyframes elb{0%,60%,100%{transform:translateY(0);opacity:.5}30%{transform:translateY(-4px);opacity:1}}" +
      "form{display:flex;gap:8px;padding:10px;border-top:1px solid #e5e7eb;background:#fff}" +
      "input{flex:1;border:1px solid #d1d5db;border-radius:8px;padding:9px 10px;font-size:13.5px;outline:none;background:#fff;color:#111827}" +
      "input:focus{border-color:#4f46e5}" +
      ".send{border:none;background:#4f46e5;color:#fff;border-radius:8px;padding:0 14px;cursor:pointer;font-weight:600;font-size:13.5px}" +
      ".send:disabled{opacity:.5;cursor:default}" +
      "</style>" +
      '<button class="bubble" title="Chat with us" aria-label="Open chat">\u{1F4AC}</button>' +
      '<div class="panel" hidden>' +
      '<div class="hdr"><span>Ask us anything</span><span class="brand">Powered by <a href="https://github.com/Vaidehi1645/easylink-ai" target="_blank" rel="noopener">EasyLink AI</a></span><button class="x" aria-label="Close">\u2715</button></div>' +
      '<div class="msgs" role="log"></div>' +
      '<form><input placeholder="Type your question\u2026" maxlength="800" autocomplete="off"><button class="send" type="submit">Send</button></form>' +
      "</div>";

    var bubble = shadow.querySelector(".bubble");
    var panel = shadow.querySelector(".panel");
    var msgs = shadow.querySelector(".msgs");
    var form = shadow.querySelector("form");
    var input = shadow.querySelector("input");
    var sendBtn = shadow.querySelector(".send");

    function add(cls, text) {
      var d = document.createElement("div");
      d.className = "m " + cls;
      d.textContent = text || "";
      msgs.appendChild(d);
      msgs.scrollTop = msgs.scrollHeight;
      return d;
    }

    bubble.addEventListener("click", function () {
      panel.hidden = !panel.hidden;
      if (!panel.hidden && !msgs.children.length) {
        add("b", "Hi! Ask me anything about this page.");
      }
      if (!panel.hidden) input.focus();
    });
    shadow.querySelector(".x").addEventListener("click", function () {
      panel.hidden = true;
    });

    var busy = false;

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var q = input.value.trim();
      if (!q || busy) return;
      input.value = "";
      add("u", q);
      busy = true;
      sendBtn.disabled = true;

      var typing = document.createElement("div");
      typing.className = "m b typing";
      typing.innerHTML = "<span></span><span></span><span></span>";
      msgs.appendChild(typing);
      msgs.scrollTop = msgs.scrollHeight;

      var answerEl = null;
      var text = "";

      fetch(base + "/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: token, question: q, history: history.slice(-4) }),
      })
        .then(function (res) {
          if (!res.ok || !res.body) {
            return res
              .json()
              .catch(function () {
                return {};
              })
              .then(function (data) {
                typing.remove();
                var err = document.createElement("div");
                err.className = "m b err";
                err.textContent = "\u26A0 " + (data.error || "Something went wrong (" + res.status + ")");
                msgs.appendChild(err);
                msgs.scrollTop = msgs.scrollHeight;
              });
          }
          var reader = res.body.getReader();
          var decoder = new TextDecoder();
          var buf = "";
          var answer = null;

          function handleEvent(line) {
            if (line.indexOf("data:") !== 0) return;
            var ev;
            try {
              ev = JSON.parse(line.slice(5).trim());
            } catch (err) {
              return;
            }
            if (ev.delta !== undefined) {
              if (!answerEl) {
                typing.remove();
                answerEl = add("b", "");
              }
              text += ev.delta;
              answerEl.textContent = text;
              msgs.scrollTop = msgs.scrollHeight;
            } else if (ev.answer) {
              answer = ev.answer;
            } else if (ev.error) {
              typing.remove();
              var errEl = document.createElement("div");
              errEl.className = "m b err";
              errEl.textContent = "\u26A0 " + ev.error;
              msgs.appendChild(errEl);
            }
          }

          function pump() {
            return reader.read().then(function (r) {
              if (r.done) {
                typing.remove();
                if (!answerEl) answerEl = add("b", "");
                if (!text) {
                  text = answer ? answer.text : "Sorry, I couldn't answer that.";
                  answerEl.textContent = text;
                }
                if (answer && answer.sources && answer.sources.length) {
                  var wrap = document.createElement("span");
                  wrap.className = "src";
                  answer.sources.slice(0, 3).forEach(function (u) {
                    var a = document.createElement("a");
                    try {
                      var parsed = new URL(u);
                      a.textContent = "\u{1F4C4} " + (parsed.pathname || "/");
                    } catch (e) {
                      a.textContent = "\u{1F4C4} source";
                    }
                    a.href = u;
                    a.target = "_blank";
                    a.rel = "noopener";
                    wrap.appendChild(a);
                  });
                  answerEl.appendChild(wrap);
                }
                if (answer) {
                  history.push({ q: q, a: answer.text });
                  if (history.length > 4) history.shift();
                }
                busy = false;
                sendBtn.disabled = false;
                input.focus();
                return;
              }
              buf += decoder.decode(r.value, { stream: true });
              var idx;
              while ((idx = buf.indexOf("\n\n")) !== -1) {
                var block = buf.slice(0, idx);
                buf = buf.slice(idx + 2);
                block.split("\n").forEach(handleEvent);
              }
              return pump();
            });
          }
          return pump();
        })
        .catch(function () {
          typing.remove();
          var errEl = document.createElement("div");
          errEl.className = "m b err";
          errEl.textContent = "\u26A0 Can't reach the assistant right now.";
          msgs.appendChild(errEl);
          msgs.scrollTop = msgs.scrollHeight;
          busy = false;
          sendBtn.disabled = false;
        });
    });
  }

  if (document.body) {
    mount();
  } else {
    document.addEventListener("DOMContentLoaded", mount);
  }
})();
