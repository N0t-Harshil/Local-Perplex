/* Local Perplex shared client code. No third-party scripts: everything runs offline. */
(function () {
  "use strict";
  var LP = (window.LP = {});

  // ── escaping & markdown ─────────────────────────────────────────────
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  LP.esc = esc;

  function inline(text, maxCite) {
    var codes = [];
    text = text.replace(/`([^`\n]+)`/g, function (_, c) { codes.push(c); return "\u0000" + (codes.length - 1) + "\u0000"; });
    text = esc(text);
    // links: only http(s) targets
    text = text.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, function (_, label, href) {
      return '<a href="' + href + '" target="_blank" rel="noopener noreferrer">' + label + "</a>";
    });
    text = text.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    text = text.replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<em>$2</em>");
    text = text.replace(/(^|[^_\w])_([^_\n]+)_(?!\w)/g, "$1<em>$2</em>");
    // citations: [1], [2][3], [1, 2]
    text = text.replace(/\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]/g, function (m, list) {
      return list.split(/\s*,\s*/).map(function (n) {
        if (maxCite && +n > maxCite) return "";
        return '<a class="cite" href="#src-' + n + '" data-n="' + n + '">' + n + "</a>";
      }).join("");
    });
    text = text.replace(/https?:\/\/[^\s<]+[^\s<.,;:!?)\]'"]/g, function (url, offset, whole) {
      var before = whole.slice(Math.max(0, offset - 6), offset);
      if (/href="$/.test(before) || /">$/.test(before)) return url;
      return '<a href="' + url + '" target="_blank" rel="noopener noreferrer">' + url + "</a>";
    });
    return text.replace(/\u0000(\d+)\u0000/g, function (_, i) { return "<code>" + esc(codes[+i]) + "</code>"; });
  }

  LP.markdown = function (src, maxCite) {
    var lines = String(src || "").replace(/\r\n?/g, "\n").split("\n");
    var out = [], i = 0, para = [];
    function flush() { if (para.length) { out.push("<p>" + inline(para.join(" "), maxCite) + "</p>"); para = []; } }
    while (i < lines.length) {
      var line = lines[i];
      var fence = line.match(/^\s*```\s*([\w+-]*)/);
      if (fence) {
        flush(); var buf = []; i++;
        while (i < lines.length && !/^\s*```/.test(lines[i])) { buf.push(lines[i]); i++; }
        i++;
        out.push('<pre><code' + (fence[1] ? ' data-lang="' + esc(fence[1]) + '"' : "") + ">" + esc(buf.join("\n")) + "</code></pre>");
        continue;
      }
      if (!line.trim()) { flush(); i++; continue; }
      var h = line.match(/^(#{1,4})\s+(.*)$/);
      if (h) { flush(); out.push("<h" + h[1].length + ">" + inline(h[2].replace(/#+\s*$/, ""), maxCite) + "</h" + h[1].length + ">"); i++; continue; }
      if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flush(); out.push("<hr>"); i++; continue; }
      if (/^\s*>/.test(line)) {
        flush(); var q = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) { q.push(lines[i].replace(/^\s*>\s?/, "")); i++; }
        out.push("<blockquote>" + LP.markdown(q.join("\n"), maxCite) + "</blockquote>");
        continue;
      }
      if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
        flush();
        var row = function (l) { return l.trim().replace(/^\||\|$/g, "").split("|").map(function (c) { return c.trim(); }); };
        var head = row(line); i += 2;
        var html = "<table><thead><tr>" + head.map(function (c) { return "<th>" + inline(c, maxCite) + "</th>"; }).join("") + "</tr></thead><tbody>";
        while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) {
          html += "<tr>" + row(lines[i]).map(function (c) { return "<td>" + inline(c, maxCite) + "</td>"; }).join("") + "</tr>"; i++;
        }
        out.push(html + "</tbody></table>");
        continue;
      }
      var li = line.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
      if (li) {
        flush();
        var ordered = /\d/.test(li[2]), items = [], baseIndent = li[1].length;
        while (i < lines.length) {
          var m = lines[i].match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
          if (m && m[1].length <= baseIndent + 1) { items.push([m[3]]); i++; continue; }
          if (m || (/^\s{2,}\S/.test(lines[i]) && items.length)) { items[items.length - 1].push(lines[i]); i++; continue; }
          break;
        }
        out.push((ordered ? "<ol>" : "<ul>") + items.map(function (it) {
          var first = inline(it[0], maxCite);
          var rest = it.slice(1).map(function (l) { return l.replace(/^\s{2,4}/, ""); }).join("\n");
          return "<li>" + first + (rest ? LP.markdown(rest, maxCite) : "") + "</li>";
        }).join("") + (ordered ? "</ol>" : "</ul>"));
        continue;
      }
      para.push(line.trim()); i++;
    }
    flush();
    return out.join("\n");
  };

  // ── toasts ─────────────────────────────────────────────────────────
  LP.toast = function (msg) {
    var box = document.getElementById("toasts");
    if (!box) return;
    var el = document.createElement("div");
    el.className = "toast"; el.setAttribute("role", "status"); el.textContent = msg;
    box.appendChild(el);
    setTimeout(function () { el.remove(); }, 2600);
  };

  // ── theme ──────────────────────────────────────────────────────────
  function storedTheme() { try { return localStorage.getItem("lp-theme"); } catch (e) { return null; } }
  LP.setTheme = function (t) {
    document.documentElement.setAttribute("data-theme", t);
    try { localStorage.setItem("lp-theme", t); } catch (e) {}
    var label = document.getElementById("theme-label");
    if (label) label.textContent = t === "dark" ? "Light theme" : "Dark theme";
  };
  LP.toggleTheme = function () {
    LP.setTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
  };

  // ── sidebar ────────────────────────────────────────────────────────
  LP.toggleSidebar = function (open) {
    var sb = document.getElementById("sidebar"), scrim = document.getElementById("scrim");
    var show = open == null ? !sb.classList.contains("open") : open;
    sb.classList.toggle("open", show); scrim.classList.toggle("show", show);
  };

  LP.loadSessions = function () {
    var list = document.getElementById("sessions");
    if (!list) return;
    var current = document.body.getAttribute("data-session") || "";
    fetch("/api/sessions").then(function (r) { return r.json(); }).then(function (data) {
      if (!data.sessions.length) { list.innerHTML = '<div class="sessions-empty">Your conversations will appear here.</div>'; return; }
      list.innerHTML = data.sessions.map(function (s) {
        return '<a class="session' + (s.session_id === current ? " current" : "") + '" href="/chat/' + s.session_id + '" data-sid="' + s.session_id + '">' +
          "<span>" + esc(s.title) + "</span>" +
          '<button type="button" data-act="rename" title="Rename" aria-label="Rename chat">✎</button>' +
          '<button type="button" data-act="delete" title="Delete" aria-label="Delete chat">×</button></a>';
      }).join("");
    }).catch(function () {});
  };

  function sessionAction(e) {
    var btn = e.target.closest("button[data-act]");
    if (!btn) return;
    e.preventDefault(); e.stopPropagation();
    var item = btn.closest(".session"), sid = item.getAttribute("data-sid");
    if (btn.dataset.act === "delete") {
      if (!confirm("Delete this conversation?")) return;
      fetch("/api/sessions/" + sid, { method: "DELETE" }).then(function () {
        LP.toast("Conversation deleted");
        if (sid === document.body.getAttribute("data-session")) location.href = "/";
        else LP.loadSessions();
      });
      return;
    }
    var span = item.querySelector("span"), old = span.textContent;
    var input = document.createElement("input");
    input.value = old; span.textContent = ""; span.appendChild(input); input.focus(); input.select();
    input.addEventListener("click", function (ev) { ev.preventDefault(); });
    input.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape") { span.textContent = old; }
      if (ev.key !== "Enter") return;
      ev.preventDefault();
      var fd = new FormData(); fd.append("title", input.value);
      fetch("/api/sessions/" + sid + "/rename", { method: "POST", body: fd }).then(function () {
        span.textContent = input.value || old; LP.toast("Conversation renamed");
      });
    });
    input.addEventListener("blur", function () { if (span.contains(input)) span.textContent = old; });
  }

  // ── composer ───────────────────────────────────────────────────────
  LP.initComposer = function (form) {
    if (!form) return;
    var ta = form.querySelector("textarea"), send = form.querySelector(".send");
    function size() { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 260) + "px"; send.disabled = !ta.value.trim(); }
    ta.addEventListener("input", size); size();
    ta.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (ta.value.trim()) form.requestSubmit(); }
    });
    form.addEventListener("submit", function () { send.disabled = true; send.innerHTML = '<span class="pulse" style="width:10px;height:10px;border-radius:50%;background:currentColor"></span>'; });
    form.querySelectorAll(".chip input[type=checkbox]").forEach(function (cb) {
      cb.addEventListener("change", function () { cb.closest(".chip").classList.toggle("on", cb.checked); });
    });
    var file = form.querySelector("input[type=file]");
    if (file) file.addEventListener("change", function () {
      var prev = form.querySelector(".image-preview");
      if (!file.files[0]) { prev.classList.remove("show"); return; }
      var r = new FileReader();
      r.onload = function (ev) { prev.querySelector("img").src = ev.target.result; prev.querySelector("span").textContent = file.files[0].name; prev.classList.add("show"); };
      r.readAsDataURL(file.files[0]);
    });
    var clear = form.querySelector(".image-preview button");
    if (clear) clear.addEventListener("click", function () { file.value = ""; form.querySelector(".image-preview").classList.remove("show"); });
  };

  LP.relTime = function (ts) {
    var d = new Date(ts * 1000), s = (Date.now() - d) / 1000;
    if (s < 60) return "just now";
    if (s < 3600) return Math.floor(s / 60) + " min ago";
    if (s < 86400) return Math.floor(s / 3600) + " h ago";
    return d.toLocaleDateString();
  };

  document.addEventListener("DOMContentLoaded", function () {
    LP.setTheme(document.documentElement.getAttribute("data-theme"));
    var list = document.getElementById("sessions");
    if (list) { list.addEventListener("click", sessionAction); LP.loadSessions(); }
    document.querySelectorAll("form.composer-form").forEach(LP.initComposer);
    document.addEventListener("keydown", function (e) {
      var t = e.target, typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
      if ((e.key === "/" && !typing) || (e.key.toLowerCase() === "k" && (e.ctrlKey || e.metaKey))) {
        var ta = document.querySelector("form.composer-form textarea");
        if (ta) { e.preventDefault(); ta.focus(); }
      }
      if (e.key === "Escape" && typing) t.blur();
    });
  });

  // apply the theme before first paint (this file loads in <head>)
  var initial = storedTheme() || (window.matchMedia && matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  document.documentElement.setAttribute("data-theme", initial);
})();
