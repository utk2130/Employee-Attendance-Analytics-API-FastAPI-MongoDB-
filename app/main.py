"""
Employee Attendance & Analytics API  (FastAPI + MongoDB)

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB  (a local .env is read, but real environment variables win)

Layout of this file, top to bottom:
  1. configuration + Mongo connection + index setup
  2. small time / rounding helpers and the business rules (R1-R5)
  3. request models and validation helpers
  4. "plans": the exact find / aggregation each list or analytics endpoint runs.
     The real endpoints and /admin/explain both use these, so explain can never
     drift away from what the endpoint really does.
  5. endpoints

A note on numbers: every reported figure that has to be rounded "half-up" is computed
with integer arithmetic (never round(), never $round, which are banker's rounding or
subject to float noise).
"""
import json
import logging
import os
import re
import threading
from calendar import monthrange
from contextlib import asynccontextmanager
from datetime import date as date_cls
from datetime import datetime, time as time_cls, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

from bson import json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, StrictInt, field_validator, model_validator
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

load_dotenv()  # does not override variables that are already set in the environment

log = logging.getLogger("attendance")

# --------------------------------------------------------------------------- #
# 1. Configuration, connection, indexes
# --------------------------------------------------------------------------- #
MONGO_URI = os.getenv("MONGO_URI" )
MONGO_DB = os.getenv("MONGO_DB")

# tz_aware=True: PyMongo hands back UTC-aware datetimes instead of naive ones, so we
# never have to guess what a datetime means. The short selection timeout keeps /health
# from hanging when Mongo is down.
client = MongoClient(MONGO_URI, tz_aware=True, serverSelectionTimeoutMS=5000)
db = client[MONGO_DB]
EMP = "employees"
LOGS = "attendance_logs"

INDEXES = [
    # employees
    (EMP, [("emp_code", ASCENDING)], {"unique": True, "name": "uniq_emp_code"}),
    (EMP, [("department", ASCENDING), ("emp_code", ASCENDING)], {"name": "dept_emp_code"}),
    (EMP, [("joined_on", ASCENDING)], {"name": "joined_on"}),
    # attendance_logs
    (LOGS, [("emp_code", ASCENDING), ("date", ASCENDING)], {"unique": True, "name": "uniq_emp_date"}),
    (LOGS, [("date", DESCENDING), ("emp_code", ASCENDING)], {"name": "date_desc_emp"}),
    (LOGS, [("status", ASCENDING), ("date", DESCENDING), ("emp_code", ASCENDING)], {"name": "status_date_emp"}),
]

_index_lock = threading.Lock()
_indexes_ready = False


def ensure_indexes() -> None:
    """Create every index; safe to call any number of times.

    A failure is logged but never stops the app from booting (for example if Mongo is
    still starting). /health calls this again, so a missed run is retried later.
    """
    global _indexes_ready
    with _index_lock:
        if _indexes_ready:
            return
        ok = True
        for coll, keys, opts in INDEXES:
            try:
                db[coll].create_index(keys, **opts)
            except PyMongoError as exc:
                ok = False
                log.warning("could not create index %s on %s: %s", opts.get("name"), coll, exc)
        _indexes_ready = ok


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_indexes()
    yield
    client.close()


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)

# --------------------------------------------------------------------------- #
# 2. Time helpers and business rules
# --------------------------------------------------------------------------- #
# <helpers>
IST = timezone(timedelta(hours=5, minutes=30))
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
PRESENCE = ("PRESENT", "WFH", "ON_DUTY")
GRACE_SECONDS = 10 * 60
MIN_OVERTIME_MINUTES = 30
HALF_DAY_BELOW_CENTI_HOURS = 450  # 4.50 hours, expressed in hundredths
MAX_SHIFT_SECONDS = 24 * 3600


def ms_to_dt(ms: int) -> datetime:
    """Epoch milliseconds -> aware UTC datetime (integer maths, no float drift)."""
    return EPOCH + timedelta(milliseconds=ms)


def dt_to_ms(dt: Optional[datetime]) -> Optional[int]:
    """Datetime -> epoch milliseconds. A naive datetime is taken to be UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - EPOCH) // timedelta(milliseconds=1)


def whole_seconds(dt: datetime) -> datetime:
    """R1: instants are cut down to whole seconds before any calculation."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.replace(microsecond=0)


def utc_now() -> datetime:
    return whole_seconds(datetime.now(timezone.utc))


def parse_hhmm(value: str) -> time_cls:
    hh, mm = value.split(":")
    return time_cls(int(hh), int(mm))


def is_overnight(shift_start: str, shift_end: str) -> bool:
    return shift_end <= shift_start  # zero-padded HH:MM strings compare correctly


def attendance_date(punch_in: datetime, shift_start: str, shift_end: str) -> str:
    """R1: the record's date is the IST calendar date of the punch-in, except that on an
    overnight shift a punch-in earlier than shift_end belongs to the previous day's shift."""
    local = punch_in.astimezone(IST)
    day = local.date()
    if is_overnight(shift_start, shift_end) and local.time() < parse_hhmm(shift_end):
        day -= timedelta(days=1)
    return day.isoformat()


def shift_window(date_str: str, shift_start: str, shift_end: str):
    """Start and end of the shift that belongs to attendance date `date_str` (both IST)."""
    day = date_cls.fromisoformat(date_str)
    start = datetime.combine(day, parse_hhmm(shift_start), tzinfo=IST)
    end = datetime.combine(day, parse_hhmm(shift_end), tzinfo=IST)
    if end <= start:
        end += timedelta(days=1)
    return start, end


def seconds_between(a: datetime, b: datetime) -> int:
    return (b - a) // timedelta(seconds=1)


def compute_late_minutes(punch_in: datetime, shift_start_dt: datetime) -> int:
    """R2: late only when strictly more than 10:00 after shift start; then whole minutes
    since shift start (not since the end of the grace period)."""
    secs = seconds_between(shift_start_dt, punch_in)
    return secs // 60 if secs > GRACE_SECONDS else 0


def compute_overtime(punch_out: datetime, shift_end_dt: datetime) -> int:
    """R3: whole minutes after shift end, counted only when that is at least 30."""
    minutes = seconds_between(shift_end_dt, punch_out) // 60
    return minutes if minutes >= MIN_OVERTIME_MINUTES else 0


def work_centi_hours(punch_in: datetime, punch_out: datetime) -> int:
    """R4 in hundredths of an hour, rounded half-up using integers only.
    seconds / 3600 hours = seconds / 36 hundredths; floor(x + 1/2) == (secs + 18) // 36."""
    return (seconds_between(punch_in, punch_out) + 18) // 36


def derive_fields(shift_start: str, shift_end: str, date_str: str, status: str,
                  punch_in: Optional[datetime], punch_out: Optional[datetime]) -> Dict[str, Any]:
    """All derived fields for one record, from final status and times (R2-R5)."""
    if status not in PRESENCE or punch_in is None:
        return {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    start_dt, end_dt = shift_window(date_str, shift_start, shift_end)
    out = {
        "work_hours": None,
        "late_minutes": compute_late_minutes(punch_in, start_dt),
        "overtime_minutes": 0,
        "half_day": False,
    }
    if punch_out is not None:
        centi = work_centi_hours(punch_in, punch_out)
        out["work_hours"] = centi / 100
        out["overtime_minutes"] = compute_overtime(punch_out, end_dt)
        out["half_day"] = centi < HALF_DAY_BELOW_CENTI_HOURS
    return out


def half_up_div(numerator: int, denominator: int) -> int:
    """round-half-up(numerator / denominator) for positive integers, exactly."""
    return (2 * numerator + denominator) // (2 * denominator)


def count_weekdays(first: date_cls, last: date_cls) -> int:
    """Mon-Fri days from `first` to `last` inclusive (R7, no holiday calendar)."""
    if last < first:
        return 0
    n = (last - first).days + 1
    full_weeks, rest = divmod(n, 7)
    total = full_weeks * 5
    weekday = first.weekday()
    for i in range(rest):
        if (weekday + i) % 7 < 5:
            total += 1
    return total
# </helpers>

# --------------------------------------------------------------------------- #
# 3. Models and validation helpers
# --------------------------------------------------------------------------- #
MIN_EPOCH_MS = 100_000_000_000
MAX_EPOCH_MS = 4_102_444_800_000
# StrictInt means 1.5, "1783312500000" and true are all rejected with 422.
EpochMs = Annotated[StrictInt, Field(ge=MIN_EPOCH_MS, le=MAX_EPOCH_MS)]

Status = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
PresenceStatus = Literal["PRESENT", "WFH", "ON_DUTY"]

DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
HHMM = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"
MONTH_PATTERN = r"^[0-9]{4}-(0[1-9]|1[0-2])$"
MAX_SKIP = 10**12  # anything beyond this is certainly an empty page; avoids int64 overflow


class EmployeeIn(BaseModel):
    emp_code: str = Field(pattern=r"^EMP[0-9]{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=HHMM)
    shift_end: str = Field(default="18:30", pattern=HHMM)
    joined_on: str

    @field_validator("joined_on")
    @classmethod
    def _real_date(cls, v: str) -> str:
        if not DATE_RE.fullmatch(v):
            raise ValueError("joined_on must look like YYYY-MM-DD")
        try:
            date_cls.fromisoformat(v)
        except ValueError:
            raise ValueError("joined_on is not a real calendar date")
        return v

    @model_validator(mode="after")
    def _shift_differs(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start and shift_end must differ")
        return self


class PunchInIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None
    status: PresenceStatus = "PRESENT"


class PunchOutIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None


class RegularizeIn(BaseModel):
    status: Optional[Status] = None
    punch_in: Optional[EpochMs] = None
    punch_out: Optional[EpochMs] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)


def invalid(where: str, field: str, msg: str, value: Any = None):
    """Raise a 422 shaped exactly like FastAPI's own validation errors."""
    raise RequestValidationError([{"type": "value_error", "loc": (where, field), "msg": msg, "input": value}])


def parse_date_param(where: str, field: str, value: str) -> date_cls:
    if not DATE_RE.fullmatch(value):
        invalid(where, field, "must be a date like YYYY-MM-DD", value)
    try:
        return date_cls.fromisoformat(value)
    except ValueError:
        invalid(where, field, "not a real calendar date", value)


def parse_month_param(month: str):
    """'2026-07' -> ('2026-07-01', '2026-07-31', first_date, last_date)."""
    if not re.fullmatch(MONTH_PATTERN, month) or int(month[:4]) < 1:
        invalid("query", "month", "must be a month like YYYY-MM", month)
    year, mon = int(month[:4]), int(month[5:7])
    first = date_cls(year, mon, 1)
    last = date_cls(year, mon, monthrange(year, mon)[1])
    return first.isoformat(), last.isoformat(), first, last


def need(name: str, value: Any, where: str = "query"):
    if value is None:
        invalid(where, name, "field required")
    return value


def get_employee(emp_code: str) -> Dict[str, Any]:
    emp = db[EMP].find_one({"emp_code": emp_code}, {"_id": 0})
    if emp is None:
        raise HTTPException(404, "employee not found")
    return emp


def shift_of(emp: Dict[str, Any]):
    return emp.get("shift_start", "09:30"), emp.get("shift_end", "18:30")


# --------------------------------------------------------------------------- #
# Serialisation: Mongo document -> API shape (no _id anywhere, instants in epoch ms)
# --------------------------------------------------------------------------- #
def serialize_employee(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "emp_code": doc["emp_code"],
        "name": doc.get("name"),
        "email": doc.get("email"),
        "department": doc.get("department"),
        "shift_start": doc.get("shift_start", "09:30"),
        "shift_end": doc.get("shift_end", "18:30"),
        "joined_on": doc.get("joined_on"),
        "created_at": dt_to_ms(doc.get("created_at")),
    }


def _history_value(field: str, value: Any) -> Any:
    return dt_to_ms(value) if field in ("punch_in", "punch_out") else value


def serialize_history(entry: Dict[str, Any]) -> Dict[str, Any]:
    changes = {
        field: {"from": _history_value(field, ch.get("from")), "to": _history_value(field, ch.get("to"))}
        for field, ch in (entry.get("changes") or {}).items()
    }
    return {"at": dt_to_ms(entry.get("at")), "by": entry.get("by"), "reason": entry.get("reason"), "changes": changes}


def serialize_record(doc: Dict[str, Any]) -> Dict[str, Any]:
    # Older documents may lack late_minutes / overtime_minutes / half_day / history.
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": dt_to_ms(doc.get("punch_in")),
        "punch_out": dt_to_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes") or 0,
        "overtime_minutes": doc.get("overtime_minutes") or 0,
        "half_day": bool(doc.get("half_day", False)),
        "history": [serialize_history(h) for h in (doc.get("history") or [])],
    }


# --------------------------------------------------------------------------- #
# 4. Plans: the query each list / analytics endpoint runs (also used by explain)
# --------------------------------------------------------------------------- #
class Plan:
    """One find() or one aggregate(), described as data, so it can be run or explained."""

    def __init__(self, collection: str, *, filter: Optional[dict] = None, sort: Optional[list] = None,
                 skip: int = 0, limit: int = 0, projection: Optional[dict] = None,
                 pipeline: Optional[list] = None):
        self.collection = collection
        self.filter = filter or {}
        self.sort = sort
        self.skip = skip
        self.limit = limit
        self.projection = projection
        self.pipeline = pipeline

    def run_find(self):
        cur = db[self.collection].find(self.filter, self.projection)
        if self.sort:
            cur = cur.sort(self.sort)
        return cur.skip(self.skip).limit(self.limit)

    def run_aggregate(self) -> List[dict]:
        return list(db[self.collection].aggregate(self.pipeline))

    def explain(self) -> dict:
        if self.pipeline is not None:
            inner = {"aggregate": self.collection, "pipeline": self.pipeline, "cursor": {}}
        else:
            inner = {"find": self.collection, "filter": self.filter}
            if self.sort:
                inner["sort"] = dict(self.sort)
            if self.projection:
                inner["projection"] = self.projection
            inner["skip"] = self.skip
            inner["limit"] = self.limit
        raw = db.command({"explain": inner, "verbosity": "executionStats"})
        return json.loads(json_util.dumps(raw))  # BSON types -> plain JSON


# ---- reusable aggregation expressions -------------------------------------- #
_SUNDAY = datetime(1970, 1, 4, tzinfo=timezone.utc)  # fallback for a malformed/missing date string
_IS_PRESENT = {"$in": ["$status", list(PRESENCE)]}
_IS_WEEKDAY = {"$lte": [{"$isoDayOfWeek": {"$dateFromString": {
    "dateString": "$date", "format": "%Y-%m-%d", "onError": _SUNDAY, "onNull": _SUNDAY}}}, 5]}
_IS_HALF_DAY = {"$eq": ["$half_day", True]}  # missing half_day -> false
_LATE_MINUTES = {"$ifNull": ["$late_minutes", 0]}
_IS_LATE = {"$gt": [_LATE_MINUTES, 0]}
# A present day is worth 2 "half-units" (a half day is 1), so sums stay exact integers.
_PRESENT_X2_ANY_DAY = {"$cond": [_IS_PRESENT, {"$cond": [_IS_HALF_DAY, 1, 2]}, 0]}
_PRESENT_X2_WEEKDAY = {"$cond": [{"$and": [_IS_PRESENT, _IS_WEEKDAY]}, {"$cond": [_IS_HALF_DAY, 1, 2]}, 0]}
_LATE_FLAG = {"$cond": [_IS_LATE, 1, 0]}
_LATE_VALUE = {"$cond": [_IS_LATE, _LATE_MINUTES, 0]}
_HAS_HOURS = {"$and": [_IS_PRESENT, {"$isNumber": "$work_hours"}]}


def _half_up_expr(sum_expr: Any, count_expr: Any, scale: int) -> dict:
    """round-half-up(sum / count) / scale, evaluated inside MongoDB with exact integers:
    floor((2*sum + count) / (2*count)) / scale."""
    return {"$divide": [{"$floor": {"$divide": [
        {"$add": [{"$multiply": [2, sum_expr]}, count_expr]},
        {"$multiply": [2, count_expr]}]}}, scale]}


def attendance_list_plan(emp_code, date_from, date_to, status, page: int, page_size: int) -> Plan:
    q: Dict[str, Any] = {}
    if emp_code is not None:
        q["emp_code"] = emp_code
    d_from = parse_date_param("query", "date_from", date_from) if date_from is not None else None
    d_to = parse_date_param("query", "date_to", date_to) if date_to is not None else None
    if d_from and d_to and d_from > d_to:
        invalid("query", "date_from", "date_from must not be after date_to", date_from)
    if d_from or d_to:
        q["date"] = {}
        if d_from:
            q["date"]["$gte"] = d_from.isoformat()
        if d_to:
            q["date"]["$lte"] = d_to.isoformat()
    if status is not None:
        q["status"] = status
    return Plan(LOGS, filter=q, sort=[("date", DESCENDING), ("emp_code", ASCENDING)],
                skip=min((page - 1) * page_size, MAX_SKIP), limit=page_size, projection={"_id": 0})


def employee_monthly_plan(emp_code: str, month: str) -> Plan:
    first, last, _, _ = parse_month_param(month)
    return Plan(LOGS, pipeline=[
        {"$match": {"emp_code": emp_code, "date": {"$gte": first, "$lte": last}}},
        {"$group": {
            "_id": None,
            "present_x2": {"$sum": _PRESENT_X2_WEEKDAY},  # R5-R7: Mon-Fri only, half day = 0.5
            "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
            "late_count": {"$sum": _LATE_FLAG},
            "total_late_minutes": {"$sum": _LATE_VALUE},
            "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
        }},
    ])


def department_summary_plan(month: str, department: Optional[str]) -> Plan:
    first, last, _, _ = parse_month_param(month)
    # Start from employees, not from logs: that is what makes people with zero logs count
    # towards headcount (R9) while people who join later are filtered out up front.
    match: Dict[str, Any] = {"joined_on": {"$lte": last}}
    if department is not None:
        match["department"] = department
    return Plan(EMP, pipeline=[
        {"$match": match},
        {"$lookup": {
            "from": LOGS, "localField": "emp_code", "foreignField": "emp_code",
            "pipeline": [
                {"$match": {"date": {"$gte": first, "$lte": last}}},
                {"$group": {
                    "_id": None,
                    "present_x2": {"$sum": _PRESENT_X2_WEEKDAY},
                    "late_count": {"$sum": _LATE_FLAG},
                    "total_late": {"$sum": _LATE_VALUE},
                    "leave": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
                    "on_duty": {"$sum": {"$cond": [{"$eq": ["$status", "ON_DUTY"]}, 1, 0]}},
                    # work_hours are stored with 2 decimals, so *100 gives exact integers
                    "wh_centi": {"$sum": {"$cond": [_HAS_HOURS, {"$round": [{"$multiply": ["$work_hours", 100]}, 0]}, 0]}},
                    "wh_n": {"$sum": {"$cond": [_HAS_HOURS, 1, 0]}},
                }},
            ],
            "as": "s",
        }},
        # keep employees that have no logs at all (empty "s") so they still count
        {"$unwind": {"path": "$s", "preserveNullAndEmptyArrays": True}},
        {"$group": {
            "_id": "$department",
            "headcount": {"$sum": 1},
            "present_x2": {"$sum": "$s.present_x2"},
            "late_count": {"$sum": "$s.late_count"},
            "total_late_minutes": {"$sum": "$s.total_late"},
            "leave_count": {"$sum": "$s.leave"},
            "on_duty_count": {"$sum": "$s.on_duty"},
            "wh_centi": {"$sum": "$s.wh_centi"},
            "wh_n": {"$sum": "$s.wh_n"},
        }},
        {"$sort": {"_id": 1}},
        {"$project": {
            "_id": 0,
            "department": "$_id",
            "headcount": 1,
            "present_days": {"$divide": ["$present_x2", 2]},
            "avg_work_hours": {"$cond": [{"$gt": ["$wh_n", 0]}, _half_up_expr("$wh_centi", "$wh_n", 100), None]},
            "late_count": 1,
            "total_late_minutes": 1,
            "leave_count": 1,
            "on_duty_count": 1,
        }},
    ])


def late_leaderboard_plan(month: str, limit: int, department: Optional[str]) -> Plan:
    first, last, _, _ = parse_month_param(month)
    emp_match: Dict[str, Any] = {}
    if department is not None:
        emp_match["department"] = department
    emp_pipeline: List[dict] = ([{"$match": emp_match}] if emp_match else []) + [
        {"$project": {"_id": 0, "name": 1, "department": 1}}]
    return Plan(LOGS, pipeline=[
        {"$match": {"date": {"$gte": first, "$lte": last}, "late_minutes": {"$gt": 0}}},
        {"$group": {"_id": "$emp_code", "total_late_minutes": {"$sum": "$late_minutes"}, "late_count": {"$sum": 1}}},
        # inner join: logs of unknown employees (and, with a department, other departments) drop out
        {"$lookup": {"from": EMP, "localField": "_id", "foreignField": "emp_code", "pipeline": emp_pipeline, "as": "e"}},
        {"$unwind": "$e"},
        # $rank is standard competition ranking: 1, 2, 2, 4. Ranking comes first, the limit after.
        {"$setWindowFields": {"sortBy": {"total_late_minutes": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {"$project": {"_id": 0, "rank": 1, "emp_code": "$_id", "name": "$e.name", "department": "$e.department",
                      "total_late_minutes": 1, "late_count": 1}},
    ])


def check_trend_range(frm: str, to: str):
    d_from = parse_date_param("query", "from", frm)
    d_to = parse_date_param("query", "to", to)
    if d_to < d_from:
        invalid("query", "to", "to must not be before from", to)
    days = (d_to - d_from).days + 1
    if days > 92:
        invalid("query", "to", "range may not exceed 92 days", to)
    return d_from, d_to, days


def department_trend_plan(department: str, frm: str, to: str) -> Plan:
    d_from, _, days = check_trend_range(frm, to)
    start = datetime(d_from.year, d_from.month, d_from.day, tzinfo=timezone.utc)
    window = {"documents": [-6, 0]}  # current row + up to 6 before it, never reaching before `from`
    return Plan(EMP, pipeline=[
        {"$match": {"department": department}},
        # per employee: their logs inside the range, already boiled down to tiny rows
        {"$lookup": {
            "from": LOGS, "localField": "emp_code", "foreignField": "emp_code",
            "pipeline": [
                {"$match": {"date": {"$gte": frm, "$lte": to}}},
                {"$project": {"_id": 0, "date": 1, "p2": _PRESENT_X2_ANY_DAY, "late": _LATE_FLAG}},
            ],
            "as": "logs",
        }},
        {"$facet": {
            "daily": [
                {"$unwind": "$logs"},
                {"$group": {"_id": "$logs.date", "p2": {"$sum": "$logs.p2"}, "late": {"$sum": "$logs.late"}}},
            ],
            "roster": [{"$group": {"_id": None, "joined": {"$push": "$joined_on"}}}],
        }},
        # the calendar itself is generated here: one entry per day from..to, weekends and gaps included
        {"$project": {
            "daily": 1,
            "joined": {"$ifNull": [{"$arrayElemAt": ["$roster.joined", 0]}, []]},
            "days": {"$map": {"input": {"$range": [0, days]}, "as": "i", "in": {"$dateToString": {
                "format": "%Y-%m-%d", "date": {"$dateAdd": {"startDate": start, "unit": "day", "amount": "$$i"}}}}}},
        }},
        {"$unwind": "$days"},
        {"$project": {
            "_id": 0,
            "date": "$days",
            "dow": {"$isoDayOfWeek": {"$dateFromString": {"dateString": "$days", "format": "%Y-%m-%d"}}},
            "headcount": {"$size": {"$filter": {"input": "$joined", "as": "j", "cond": {"$lte": ["$$j", "$days"]}}}},  # R9
            "d": {"$arrayElemAt": [{"$filter": {"input": "$daily", "as": "x", "cond": {"$eq": ["$$x._id", "$days"]}}}, 0]},
        }},
        {"$project": {
            "date": 1, "headcount": 1,
            "is_working_day": {"$lte": ["$dow", 5]},
            "p2": {"$ifNull": ["$d.p2", 0]},
            "late_count": {"$ifNull": ["$d.late", 0]},
        }},
        # rate in ten-thousandths, half-up with integers; null off working days / with no headcount
        {"$project": {
            "date": 1, "headcount": 1, "is_working_day": 1, "p2": 1, "late_count": 1,
            "rate_k": {"$cond": [{"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
                                 {"$floor": {"$divide": [{"$add": [{"$multiply": [10000, "$p2"]}, "$headcount"]},
                                                         {"$multiply": [2, "$headcount"]}]}}, None]},
        }},
        {"$setWindowFields": {"sortBy": {"date": 1}, "output": {
            "win_sum": {"$sum": "$rate_k", "window": window},
            "win_n": {"$sum": {"$cond": [{"$isNumber": "$rate_k"}, 1, 0]}, "window": window},
        }}},
        {"$sort": {"date": 1}},
        {"$project": {
            "_id": 0, "date": 1, "is_working_day": 1, "headcount": 1,
            "present_count": {"$divide": ["$p2", 2]},
            "late_count": 1,
            "attendance_rate": {"$cond": [{"$isNumber": "$rate_k"}, {"$divide": ["$rate_k", 10000]}, None]},
            "moving_avg_7d": {"$cond": [{"$gt": ["$win_n", 0]}, _half_up_expr("$win_sum", "$win_n", 10000), None]},
        }},
    ])


# --------------------------------------------------------------------------- #
# 5. Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
    except PyMongoError:
        raise HTTPException(503, "MongoDB is not reachable")
    ensure_indexes()
    return {"status": "ok"}


# ---- employees ------------------------------------------------------------- #
@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = ms_to_dt(dt_to_ms(utc_now()))
    try:
        # no "look first": the unique index on emp_code is what decides, so a double
        # submit can never create two employees.
        db[EMP].insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return serialize_employee(doc)


@app.get("/employees")
def list_employees(department: Optional[str] = None,
                   page: int = Query(1, ge=1),
                   page_size: int = Query(20, ge=1, le=100)):
    q: Dict[str, Any] = {}
    if department is not None:
        q["department"] = department
    skip = min((page - 1) * page_size, MAX_SKIP)
    total = db[EMP].count_documents(q)
    docs = db[EMP].find(q, {"_id": 0}).sort("emp_code", ASCENDING).skip(skip).limit(page_size)
    return {"items": [serialize_employee(d) for d in docs], "total": total, "page": page, "page_size": page_size}


# ---- attendance ------------------------------------------------------------ #
@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = get_employee(body.emp_code)
    shift_start, shift_end = shift_of(emp)
    ts = whole_seconds(ms_to_dt(body.punched_at)) if body.punched_at is not None else utc_now()
    day = attendance_date(ts, shift_start, shift_end)
    doc = {"emp_code": body.emp_code, "date": day, "status": body.status, "punch_in": ts, "punch_out": None,
           **derive_fields(shift_start, shift_end, day, body.status, ts, None), "history": []}
    try:
        # Race safety lives in the unique (emp_code, date) index: of two simultaneous
        # requests exactly one insert succeeds and the other one raises DuplicateKeyError.
        db[LOGS].insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return serialize_record(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = get_employee(body.emp_code)
    shift_start, shift_end = shift_of(emp)
    ts = whole_seconds(ms_to_dt(body.punched_at)) if body.punched_at is not None else utc_now()

    # most recent record that started at or before the punch-out (the date sort rides the
    # unique index backwards; records without a punch_in never satisfy $lte on a date)
    rec = db[LOGS].find_one({"emp_code": body.emp_code, "punch_in": {"$lte": ts}}, {"_id": 0},
                            sort=[("date", DESCENDING)])
    if rec is None:
        raise HTTPException(404, "no punch-in found for this employee")
    if rec.get("punch_out") is not None:
        raise HTTPException(409, "already punched out")

    pin = whole_seconds(rec["punch_in"])
    if ts <= pin:
        invalid("body", "punched_at", "punched_at must be after the punch-in", body.punched_at)
    if seconds_between(pin, ts) > MAX_SHIFT_SECONDS:
        invalid("body", "punched_at", "punched_at is more than 24 hours after the punch-in", body.punched_at)

    d = derive_fields(shift_start, shift_end, rec["date"], rec["status"], pin, ts)
    # Atomic close: only succeeds while punch_out is still empty, so of two simultaneous
    # punch-outs exactly one gets the document and the other falls through to 409.
    updated = db[LOGS].find_one_and_update(
        {"emp_code": rec["emp_code"], "date": rec["date"], "punch_out": None, "punch_in": rec["punch_in"]},
        {"$set": {"punch_out": ts, "work_hours": d["work_hours"],
                  "overtime_minutes": d["overtime_minutes"], "half_day": d["half_day"]}},
        return_document=ReturnDocument.AFTER, projection={"_id": 0})
    if updated is None:
        raise HTTPException(409, "already punched out")
    return serialize_record(updated)


@app.get("/attendance")
def list_attendance(emp_code: Optional[str] = None,
                    date_from: Optional[str] = None,
                    date_to: Optional[str] = None,
                    status: Optional[Status] = None,
                    page: int = Query(1, ge=1),
                    page_size: int = Query(20, ge=1, le=100)):
    plan = attendance_list_plan(emp_code, date_from, date_to, status, page, page_size)
    # an unfiltered total is served from collection metadata instead of scanning everything
    total = db[LOGS].count_documents(plan.filter) if plan.filter else db[LOGS].estimated_document_count()
    items = [serialize_record(d) for d in plan.run_find()]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


FIELD_ORDER = ("status", "punch_in", "punch_out", "work_hours", "late_minutes", "overtime_minutes", "half_day")


def _comparable(field: str, value: Any) -> Any:
    if field in ("punch_in", "punch_out"):
        return dt_to_ms(value)
    if field == "work_hours":
        return None if value is None else round(float(value), 2)
    return value


@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(emp_code: str, date: str, body: RegularizeIn):
    parse_date_param("path", "date", date)
    emp = get_employee(emp_code)
    shift_start, shift_end = shift_of(emp)
    rec = db[LOGS].find_one({"emp_code": emp_code, "date": date}, {"_id": 0})
    if rec is None:
        raise HTTPException(404, "no attendance record for that date")

    old_status = rec["status"]
    old_in = whole_seconds(rec["punch_in"]) if rec.get("punch_in") else None
    old_out = whole_seconds(rec["punch_out"]) if rec.get("punch_out") else None
    new_status = body.status or old_status

    if new_status in ("ABSENT", "LEAVE"):
        # these statuses clear the punch times, so sending times along is a contradiction
        if body.punch_in is not None:
            invalid("body", "punch_in", "cannot be set together with status ABSENT/LEAVE", body.punch_in)
        if body.punch_out is not None:
            invalid("body", "punch_out", "cannot be set together with status ABSENT/LEAVE", body.punch_out)
        new_in = new_out = None
    else:
        new_in = whole_seconds(ms_to_dt(body.punch_in)) if body.punch_in is not None else old_in
        new_out = whole_seconds(ms_to_dt(body.punch_out)) if body.punch_out is not None else old_out
        if new_in is None:
            invalid("body", "punch_in", "a presence status needs a punch_in", None)
        if attendance_date(new_in, shift_start, shift_end) != date:
            invalid("body", "punch_in", "punch_in must stay on the record's attendance date", body.punch_in)
        if new_out is not None:
            if new_out <= new_in:
                invalid("body", "punch_out", "punch_out must be after punch_in", body.punch_out)
            if seconds_between(new_in, new_out) > MAX_SHIFT_SECONDS:
                invalid("body", "punch_out", "punch_out is more than 24 hours after punch_in", body.punch_out)

    if new_status == old_status and new_in == old_in and new_out == old_out:
        invalid("body", "status", "this request changes nothing", None)

    new_values = {"status": new_status, "punch_in": new_in, "punch_out": new_out,
                  **derive_fields(shift_start, shift_end, date, new_status, new_in, new_out)}
    old_values = {"status": old_status, "punch_in": old_in, "punch_out": old_out,
                  "work_hours": rec.get("work_hours"), "late_minutes": rec.get("late_minutes") or 0,
                  "overtime_minutes": rec.get("overtime_minutes") or 0, "half_day": bool(rec.get("half_day", False))}
    changes = {f: {"from": old_values[f], "to": new_values[f]} for f in FIELD_ORDER
               if _comparable(f, old_values[f]) != _comparable(f, new_values[f])}

    entry = {"at": utc_now(), "by": body.regularized_by, "reason": body.reason, "changes": changes}
    update: Dict[str, Any] = {"$set": {f: new_values[f] for f in changes}}
    history = rec.get("history")
    if isinstance(history, list):
        update["$push"] = {"history": entry}
        history_guard: Dict[str, Any] = {"$size": len(history)}
    else:  # legacy document without a usable history array
        update["$set"]["history"] = [entry]
        history_guard = {"$not": {"$type": "array"}}

    # Optimistic concurrency: the update only applies if the record is still exactly as we
    # read it (same status, times and number of history entries). If somebody got in first
    # we answer 409 instead of overwriting their change or dropping their history entry.
    updated = db[LOGS].find_one_and_update(
        {"emp_code": emp_code, "date": date, "status": old_status, "punch_in": rec.get("punch_in"),
         "punch_out": rec.get("punch_out"), "history": history_guard},
        update, return_document=ReturnDocument.AFTER, projection={"_id": 0})
    if updated is None:
        raise HTTPException(409, "the record was changed by someone else, please retry")
    return serialize_record(updated)


# ---- analytics ------------------------------------------------------------- #
@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str = Query(pattern=MONTH_PATTERN)):
    first, last, first_d, last_d = parse_month_param(month)
    emp = get_employee(emp_code)
    rows = employee_monthly_plan(emp_code, month).run_aggregate()
    agg = rows[0] if rows else {}
    present_x2 = int(agg.get("present_x2", 0))

    joined = date_cls.fromisoformat(emp["joined_on"])
    working_days = count_weekdays(max(first_d, joined), last_d)  # R7: mid-month joiners count from joined_on
    pct = None
    if working_days > 0:
        # present/working*100 to 2 decimals, half-up: floor((10000*x2 + W) / (2W)) / 100
        pct = half_up_div(5000 * present_x2, working_days) / 100
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present_x2 / 2,
        "leave_days": int(agg.get("leave_days", 0)),
        "late_count": int(agg.get("late_count", 0)),
        "total_late_minutes": int(agg.get("total_late_minutes", 0)),
        "total_overtime_minutes": int(agg.get("total_overtime_minutes", 0)),
        "attendance_pct": pct,
    }


@app.get("/analytics/departments/summary")
def department_summary(month: str = Query(pattern=MONTH_PATTERN), department: Optional[str] = None):
    rows = department_summary_plan(month, department).run_aggregate()
    items = [{
        "department": r["department"],
        "headcount": r["headcount"],
        "present_days": r["present_days"],
        "avg_work_hours": r.get("avg_work_hours"),
        "late_count": r["late_count"],
        "total_late_minutes": r["total_late_minutes"],
        "leave_count": r["leave_count"],
        "on_duty_count": r["on_duty_count"],
    } for r in rows]
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(month: str = Query(pattern=MONTH_PATTERN),
                     limit: int = Query(10, ge=1, le=50),
                     department: Optional[str] = None):
    rows = late_leaderboard_plan(month, limit, department).run_aggregate()
    return {"month": month, "items": rows}


@app.get("/analytics/departments/{department}/trend")
def department_trend(department: str,
                     frm: str = Query(alias="from"),
                     to: str = Query()):
    plan = department_trend_plan(department, frm, to)  # validates the range first (422)
    if db[EMP].find_one({"department": department}, {"_id": 1}) is None:
        raise HTTPException(404, "unknown department")
    rows = plan.run_aggregate()
    items = [{
        "date": r["date"],
        "is_working_day": r["is_working_day"],
        "headcount": r["headcount"],
        "present_count": r["present_count"],
        "late_count": r["late_count"],
        "attendance_rate": r.get("attendance_rate"),
        "moving_avg_7d": r.get("moving_avg_7d"),
    } for r in rows]
    return {"department": department, "items": items}


# ---- admin ----------------------------------------------------------------- #
@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: Literal["attendance_list", "employee_monthly", "department_summary", "late_leaderboard", "department_trend"],
    emp_code: Optional[str] = None,
    month: Optional[str] = Query(None, pattern=MONTH_PATTERN),
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    frm: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    # Same builders as the real endpoints, so the plan we show is the plan that runs.
    if endpoint == "attendance_list":
        plan = attendance_list_plan(emp_code, date_from, date_to, status, page, page_size)
    elif endpoint == "employee_monthly":
        plan = employee_monthly_plan(need("emp_code", emp_code), need("month", month))
    elif endpoint == "department_summary":
        plan = department_summary_plan(need("month", month), department)
    elif endpoint == "late_leaderboard":
        plan = late_leaderboard_plan(need("month", month), limit, department)
    else:
        plan = department_trend_plan(need("department", department), need("from", frm), need("to", to))
    return {"endpoint": endpoint, "collection": plan.collection, "explain": plan.explain()}
