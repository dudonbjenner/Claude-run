#!/usr/bin/env python3
"""Flask web UI for the lead scraper with real-time SSE streaming."""

import io
import csv
import json
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, fields

from flask import Flask, render_template, request, jsonify, Response, stream_with_context

from lead_scraper import LeadScraper, Lead

app = Flask(__name__)

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _normalise_url(url: str) -> str:
    url = url.strip()
    if url and not url.startswith("http"):
        url = "https://" + url
    return url


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scrape", methods=["POST"])
def start_scrape():
    body = request.get_json(force=True)
    raw = body.get("urls", "")
    urls = [
        _normalise_url(line)
        for line in raw.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not urls:
        return jsonify({"error": "No valid URLs provided"}), 400

    workers   = max(1, int(body.get("workers",   1)))
    delay     = max(0.0, float(body.get("delay",   1.5)))
    timeout   = max(1,   int(body.get("timeout",  10)))
    max_pages = max(1,   int(body.get("max_pages", 4)))

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"results": [], "total": len(urls), "done": False}

    def _run():
        scraper = LeadScraper(delay=delay, timeout=timeout, max_pages_per_site=max_pages)
        if workers == 1:
            for url in urls:
                try:
                    lead = scraper.scrape(url)
                except Exception:
                    lead = Lead(url=url, status="error")
                with _jobs_lock:
                    _jobs[job_id]["results"].append(asdict(lead))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(scraper.scrape, url): url for url in urls}
                for fut in as_completed(futs):
                    try:
                        lead = fut.result()
                    except Exception:
                        lead = Lead(url=futs[fut], status="error")
                    with _jobs_lock:
                        _jobs[job_id]["results"].append(asdict(lead))
        with _jobs_lock:
            _jobs[job_id]["done"] = True

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "total": len(urls)})


@app.route("/api/stream/<job_id>")
def stream(job_id):
    def generate():
        sent = 0
        while True:
            with _jobs_lock:
                job = _jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                return
            results = job["results"]
            while sent < len(results):
                payload = {"lead": results[sent], "index": sent + 1, "total": job["total"]}
                yield f"data: {json.dumps(payload)}\n\n"
                sent += 1
            if job["done"] and sent >= len(results):
                yield f"data: {json.dumps({'done': True, 'total': sent})}\n\n"
                return
            time.sleep(0.25)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/download/<job_id>")
def download(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return "Job not found", 404
    col_names = [f.name for f in fields(Lead)]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=col_names)
    writer.writeheader()
    for row in job["results"]:
        writer.writerow(row)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=leads.csv"},
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
