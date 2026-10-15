# DECISIONS.md

1. **Indexes.** Logs: a unique `(emp_code, date)`: it is the natural key, makes punch-in race-safe, and serves per-employee queries and the joins. `(date desc, emp_code)` gives the list its sort order and serves date-range scans. `(status, date desc, emp_code)` serves status filters. Employees: unique `emp_code`, `(department, emp_code)`, `joined_on` for headcount. I rejected a `punch_in` index for punch-out: I walk the unique index backwards by date instead, which avoids an extra index on every write.

2. **Punch-in race.** Both requests work out the same (emp_code, date) and call `insert_one` with no read first. The unique index lets exactly one in; the other gets `DuplicateKeyError`, which I return as 409.

3. **Ties.** Ranking uses `$rank` (1, 2, 2, 4) and `limit` filters `rank <= limit` afterwards, so everyone tied at the cutoff comes back, even if that exceeds `limit`. Ties are ordered by emp_code.

4. **Headcount.** The pipeline starts from `employees` (joined on or before month end) and looks logs up per person, keeping people whose lookup is empty. They count once, with zero days.

5. **100x data.** I would stop aggregating raw logs for every analytics call and keep a monthly roll-up collection, updated on each write.
