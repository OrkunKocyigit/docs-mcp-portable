"use strict";

// Every value from the server is rendered with textContent; nothing is parsed as HTML.
const token = document.querySelector('meta[name="csrf-token"]').content;
const $ = (id) => document.getElementById(id);

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "className") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child);
  return node;
}

async function api(method, path, body) {
  const options = { method, headers: { "X-CSRF-Token": token } };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const res = await fetch(path, options);
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    const detail = data && (data.error || (Array.isArray(data.detail) && data.detail.map((d) => d.msg).join("; ")));
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return data;
}

function when(ts) {
  return ts ? new Date(ts * 1000).toLocaleString() : "never";
}

function setStatus(node, text, isError = false) {
  node.textContent = text;
  node.classList.toggle("error", isError);
}

// ------------------------------------------------------------------ add

function currentKind() {
  return document.querySelector('input[name="kind"]:checked').value;
}

for (const radio of document.querySelectorAll('input[name="kind"]')) {
  radio.addEventListener("change", () => {
    for (const row of document.querySelectorAll("[data-kind]")) row.hidden = row.dataset.kind !== currentKind();
  });
}

$("add-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const kind = currentKind();
  const body = kind === "local"
    ? { kind, path: $("add-path").value.trim() }
    : { kind, url: $("add-url").value.trim(), ref: $("add-ref").value.trim() || null };
  setStatus($("add-status"), "Adding...");
  try {
    const out = await api("POST", "/api/projects", body);
    setStatus($("add-status"), `Added ${out.project.name}; ${out.job.action} queued.`);
    $("add-path").value = $("add-url").value = $("add-ref").value = "";
    refresh();
  } catch (err) {
    setStatus($("add-status"), err.message, true);
  }
});

// ------------------------------------------------------------------ projects

let projects = [];

function indexCell(p) {
  if (p.status) {
    const langs = Object.entries(p.status.languages || {}).sort((a, b) => b[1] - a[1]).slice(0, 4)
      .map(([name, n]) => `${name} ${n}`).join(", ");
    return `${p.status.files} files, ${p.status.chunks} chunks${langs ? ` (${langs})` : ""}\nindexed ${when(p.indexed_at)}`;
  }
  return p.indexed_at ? `indexed ${when(p.indexed_at)}` : "not indexed";
}

function stateCell(p) {
  if (p.job) {
    const step = p.job.progress && p.job.progress.step ? `: ${p.job.progress.step}` : "";
    const files = p.job.progress && p.job.progress.files !== undefined ? ` (${p.job.progress.files} files)` : "";
    return el("span", { className: "busy", text: `${p.job.state}${step}${files}` });
  }
  if (p.last_error) return el("span", { className: "error", text: p.last_error });
  return el("span", { className: "ok", text: p.indexed_at ? "ready" : "-" });
}

function renderProjects() {
  const body = $("projects").querySelector("tbody");
  body.replaceChildren();
  for (const p of projects) {
    const source = p.kind === "git" ? `${p.url}${p.ref ? ` @ ${p.ref}` : ""}` : p.path;
    const reindex = el("button", {
      text: p.kind === "git" ? "Pull + index" : "Re-index",
      onclick: async () => {
        try { await api("POST", `/api/projects/${encodeURIComponent(p.id)}/index`); refresh(); }
        catch (err) { alert(err.message); }
      },
    });
    const remove = el("button", {
      className: "danger",
      text: "Remove",
      onclick: async () => {
        const what = p.kind === "git" ? "its clone and its index" : "its index (your files are not touched)";
        if (!confirm(`Remove ${p.name}? This deletes ${what}.`)) return;
        try { await api("DELETE", `/api/projects/${encodeURIComponent(p.id)}`); refresh(); }
        catch (err) { alert(err.message); }
      },
    });
    if (p.job) reindex.disabled = remove.disabled = true;
    body.append(el("tr", {},
      el("td", { text: p.name }),
      el("td", { className: "source", text: source }),
      el("td", { className: "pre", text: indexCell(p) }),
      el("td", {}, stateCell(p)),
      el("td", { className: "actions" }, reindex, remove)));
  }
  $("projects").hidden = projects.length === 0;
  $("projects-empty").hidden = projects.length !== 0;

  const select = $("search-project");
  const chosen = select.value;
  select.replaceChildren(...projects.map((p) => el("option", { value: p.id, text: p.name })));
  if (projects.some((p) => p.id === chosen)) select.value = chosen;
}

function renderJobs(jobs) {
  const project = Object.fromEntries(projects.map((p) => [p.id, p.name]));
  $("jobs").replaceChildren(...jobs.slice(0, 15).map((j) => el("li", { className: j.state },
    el("span", { className: "job-state", text: j.state }),
    el("span", { text: ` ${project[j.project_id] || j.project_id}: ${j.action}` }),
    el("span", { className: "muted", text: j.message ? ` - ${j.message}` : "" }))));
}

let timer = null;
async function refresh() {
  clearTimeout(timer);
  let busy = false;
  try {
    const [p, j] = await Promise.all([api("GET", "/api/projects"), api("GET", "/api/jobs")]);
    projects = p.projects;
    renderProjects();
    renderJobs(j.jobs);
    busy = j.jobs.some((job) => job.state === "queued" || job.state === "running");
  } catch (err) {
    setStatus($("add-status"), `Cannot reach the server: ${err.message}`, true);
  }
  timer = setTimeout(refresh, busy ? 1500 : 10000);
}

// ------------------------------------------------------------------ search

function list(value) {
  const items = value.split(",").map((s) => s.trim()).filter(Boolean);
  return items.length ? items : null;
}

$("search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const body = {
    project_id: $("search-project").value,
    query: $("search-query").value.trim(),
    limit: Number($("search-limit").value) || 10,
    languages: list($("search-languages").value.toLowerCase()),
    paths: list($("search-paths").value),
  };
  if (!body.project_id || !body.query) return;
  setStatus($("search-status"), "Searching...");
  $("results").replaceChildren();
  try {
    const out = await api("POST", "/api/search", body);
    setStatus($("search-status"), `${out.results.length} result(s).`);
    $("results").replaceChildren(...out.results.map((r) => el("li", {},
      el("div", { className: "result-head" },
        el("span", { className: "path", text: `${r.file_path}:${r.start_line}-${r.end_line}` }),
        el("span", { className: "muted", text: ` ${r.language} - score ${r.score}` })),
      el("pre", { text: r.content }))));
  } catch (err) {
    setStatus($("search-status"), err.message, true);
  }
});

// ------------------------------------------------------------------ MCP

$("mcp-config").textContent = JSON.stringify(
  { mcp: { "code-search": { type: "remote", url: $("mcp-url").textContent } } }, null, 2);

refresh();
