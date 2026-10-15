# Employee Attendance & Analytics API

FastAPI + MongoDB. All application code is in `app/main.py`.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MONGO_URI="mongodb://localhost:27017"   # or your Atlas URI
export MONGO_DB="attendance_db"
uvicorn app.main:app --port 8000
```
A local `.env` with those two variables also works; real environment variables take priority. Indexes are created automatically at startup. MongoDB 6.0 or newer is needed (`$setWindowFields`, `$dateAdd`).

## Things worth knowing
- **Rounding:** every half-up figure is computed with integer arithmetic, never `round()` or `$round`.
- **Moving average:** `moving_avg_7d` averages the *rounded* `attendance_rate` values shown in the response (contract wording: "mean of the non-null `attendance_rate` values").
- **PATCH conflicts:** concurrent corrections use optimistic checking; the loser gets 409 and nothing is lost.
- **Explain:** `/admin/explain/{endpoint}` uses the same query builders as the real endpoints.

## Status
All endpoints in the contract are implemented. The business-rule helpers (R1-R5, rounding, working days) were checked on their own against the contract examples and the sample documents. I have not been able to run the aggregation pipelines against a live MongoDB with the large dataset, so please treat the analytics endpoints as the part to verify first.
