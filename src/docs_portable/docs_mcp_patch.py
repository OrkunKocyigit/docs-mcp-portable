"""Adds a rerank stage to docs-mcp-server's hybrid search (arabold/docs-mcp-server#472).

``DocumentStore.findByContent`` sorts the vector + full-text candidates by RRF and cuts them to ``limit``.
The patch hands the top ``DOCS_PORTABLE_RERANK_CANDIDATES`` of that list to the local /v1/rerank endpoint
first, before results are grouped and assembled, so the reranker sees single chunks. Without the env vars,
or when the endpoint fails, the original RRF order is kept.

The patch is an exact text replacement of the bundled ``dist`` code and refuses to touch versions it
was not written for.
"""

from __future__ import annotations

import os
from pathlib import Path

SUPPORTED_VERSIONS = ("3.4.1",)
MARKER = "/* docs-portable:rerank v1 */"
METHOD_ANCHOR = "async findByContent(library, version, query, limit) {"

REPLACEMENTS = (
    (
        "return this.assignRanks(rawResults).sort((a, b) => b.rrf_score - a.rrf_score || compareTies(a, b))"
        ".slice(0, limit).map((row) => {",
        "return (await __docsPortableRerank(query, this.assignRanks(rawResults).sort((a, b) => b.rrf_score"
        " - a.rrf_score || compareTies(a, b)), limit)).map((row) => {",
    ),
    # Results are re-sorted by `score` after assembly, so it has to carry the rerank order.
    ("score: row.rrf_score,", "score: row.rerank_score ?? row.rrf_score,"),
)

HELPER = r"""
async function __docsPortableRerank(query, ranked, limit) {
	const fallback = () => ranked.slice(0, limit);
	const endpoint = process.env.DOCS_PORTABLE_RERANK_URL;
	const depth = Number.parseInt(process.env.DOCS_PORTABLE_RERANK_CANDIDATES ?? "", 10);
	if (!endpoint || !(depth > 0) || ranked.length < 2) return fallback();
	const timeoutMs = Number.parseInt(process.env.DOCS_PORTABLE_RERANK_TIMEOUT_MS ?? "", 10) || 30000;
	const warn = (message) => (typeof logger !== "undefined" ? logger.warn(message) : console.warn(message));
	try {
		const url = new URL(endpoint);
		if (url.protocol !== "http:" || !["127.0.0.1", "localhost", "[::1]"].includes(url.hostname)) {
			throw new Error("DOCS_PORTABLE_RERANK_URL must be a loopback http URL");
		}
		const pool = ranked.slice(0, Math.max(limit, depth));
		const response = await fetch(url, {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify({
				model: process.env.DOCS_PORTABLE_RERANK_MODEL || undefined,
				query,
				documents: pool.map((row) => String(row.content ?? "")),
				top_n: Math.min(limit, pool.length),
				return_documents: false
			}),
			signal: AbortSignal.timeout(timeoutMs)
		});
		if (!response.ok) throw new Error(`HTTP ${response.status}`);
		const body = await response.json();
		const items = Array.isArray(body?.results) ? body.results : [];
		const seen = new Set();
		const reranked = [];
		for (const item of items) {
			const index = item?.index;
			const score = item?.relevance_score;
			if (!Number.isInteger(index) || index < 0 || index >= pool.length || seen.has(index)) throw new Error("invalid result index");
			if (typeof score !== "number" || !Number.isFinite(score)) throw new Error("invalid relevance_score");
			seen.add(index);
			reranked.push({ ...pool[index], rerank_score: score });
		}
		if (reranked.length !== Math.min(limit, pool.length)) throw new Error("unexpected number of results");
		return reranked;
	} catch (error) {
		warn(`docs-portable: rerank failed, using hybrid ranking (${error?.message ?? error})`);
		return fallback();
	}
}
"""


class PatchError(RuntimeError):
    pass


def patch_source(source: str) -> str:
    """Return the patched bundle; unchanged if already patched. Raises PatchError on any mismatch."""
    if MARKER in source:
        return source
    for old, new in REPLACEMENTS:
        count = source.count(old)
        if count != 1:
            raise PatchError(f"expected exactly one occurrence of {old[:60]!r}..., found {count}")
        source = source.replace(old, new)
    return f"{source.rstrip()}\n{MARKER}{HELPER}"


def find_bundle(package_dir: Path) -> Path:
    matches = [p for p in sorted((package_dir / "dist").glob("*.js")) if METHOD_ANCHOR in _read(p)]
    if len(matches) != 1:
        raise PatchError(f"expected one dist bundle defining findByContent, found {len(matches)}")
    return matches[0]


def apply(package_dir: Path, version: str | None) -> bool:
    """Patch the installed package in place. Returns True if the file changed."""
    if version not in SUPPORTED_VERSIONS:
        raise PatchError(
            f"the rerank patch supports docs-mcp-server {', '.join(SUPPORTED_VERSIONS)}, found {version}"
        )
    target = find_bundle(package_dir)
    source = _read(target)
    patched = patch_source(source)
    if patched == source:
        return False
    tmp = target.with_name(target.name + ".docs-portable.tmp")
    tmp.write_bytes(patched.encode("utf-8"))
    os.replace(tmp, target)
    return True


def _read(path: Path) -> str:
    return path.read_bytes().decode("utf-8")
