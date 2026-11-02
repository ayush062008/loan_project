import json
import re
import shutil
import uuid
from pathlib import Path

import joblib
from django.conf import settings
from django.http import Http404, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from .forms import UploadForm
from .ml_pipeline import load_dataset, predict_one, raw_summary, run_pipeline, suggest_target

RUNS = Path(settings.MEDIA_ROOT) / "runs"
SAMPLE = Path(settings.BASE_DIR) / "data" / "loan_approval_sample.csv"
_bundle_cache: dict = {}


# ---- helpers ---------------------------------------------------------------
def _run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):       # blocks path traversal
        raise Http404
    d = RUNS / run_id
    if not d.is_dir():
        raise Http404("This session has expired. Upload the dataset again.")
    return d


def _raw_file(d: Path) -> Path:
    return next(d.glob("raw.*"))


def _result(run_id: str):
    d = _run_dir(run_id)
    f = d / "result.json"
    return json.loads(f.read_text()) if f.exists() else None


def _ctx(run_id, res, tab, **extra):
    return {"run_id": run_id, "r": res, "tab": tab, "media": f"{settings.MEDIA_URL}runs/{run_id}/", **extra}


def _new_run(src: Path, name: str) -> str:
    run_id = uuid.uuid4().hex
    d = RUNS / run_id
    d.mkdir(parents=True)
    shutil.copy(src, d / f"raw{Path(name).suffix.lower()}")
    (d / "name.txt").write_text(name)
    return run_id


# ---- views -----------------------------------------------------------------
def upload(request):
    error = None
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            f = form.cleaned_data["file"]
            run_id = uuid.uuid4().hex
            d = RUNS / run_id
            d.mkdir(parents=True)
            dest = d / f"raw{Path(f.name).suffix.lower()}"
            with open(dest, "wb") as out:
                for chunk in f.chunks():
                    out.write(chunk)
            (d / "name.txt").write_text(f.name)
            try:
                load_dataset(dest)
            except Exception as exc:
                shutil.rmtree(d, ignore_errors=True)
                error = f"Could not read that file: {exc}"
            else:
                return redirect("dataset", run_id=run_id)
        else:
            error = form.errors["file"][0]
    return render(request, "mining/upload.html", {"error": error, "has_sample": SAMPLE.exists(), "tab": "upload"})


def use_sample(request):
    return redirect("dataset", run_id=_new_run(SAMPLE, SAMPLE.name))


def dataset(request, run_id):
    d = _run_dir(run_id)
    df = load_dataset(_raw_file(d))
    name = (d / "name.txt").read_text()
    error = None
    if request.method == "POST":
        target = request.POST.get("target")
        try:
            if target not in df.columns:
                raise ValueError("Choose a target column from the list.")
            run_pipeline(df, target, d)
            _bundle_cache.pop(run_id, None)
            return redirect("summary", run_id=run_id)
        except Exception as exc:
            error = str(exc)
    return render(request, "mining/dataset.html", {
        "run_id": run_id, "name": name, "s": raw_summary(df), "columns": list(df.columns),
        "target": request.POST.get("target") or suggest_target(df), "error": error, "tab": "dataset"})


def summary(request, run_id):
    res = _result(run_id)
    if not res:
        return redirect("dataset", run_id=run_id)
    return render(request, "mining/summary.html", _ctx(run_id, res, "summary"))


def dashboard(request, run_id):
    res = _result(run_id)
    if not res:
        return redirect("dataset", run_id=run_id)
    return render(request, "mining/dashboard.html", _ctx(run_id, res, "dashboard"))


def predict(request, run_id):
    res = _result(run_id)
    if not res:
        return redirect("dataset", run_id=run_id)
    return render(request, "mining/predict.html", _ctx(run_id, res, "predict"))


@require_POST
def predict_api(request, run_id):
    d = _run_dir(run_id)
    if run_id not in _bundle_cache:
        if not (d / "model.joblib").exists():
            return JsonResponse({"error": "Train the model first."}, status=400)
        _bundle_cache[run_id] = joblib.load(d / "model.joblib")
    try:
        values = json.loads(request.body or "{}")
        return JsonResponse(predict_one(_bundle_cache[run_id], values))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
