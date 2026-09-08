# SCHEMA.md — the log contract

**Status:** proposed by P3. **Frozen**

Every folder in this repo reads from this one table. Once it is frozen it
does not change — a rename on Day 2 breaks three people's work at once.

---

## How this document was derived

We did not start from "what can the API easily record?" We started from
"what will each person need to look up, and what must be in front of them
to look it up?" Every column below survived two filters:

- **Exactly one writer.** Two writers means two people disagreeing about
  what a column means, discovered on Day 3.
- **At least one named reader.** A column nobody can claim is a column
  nobody can interpret later.

If you cannot find your need in this table, say so before you sign.

---

## Table: `requests`

One row per API call.

| # | Column | Type | Null? | Written by | Read by, and for what |
|---|--------|------|-------|-----------|----------------------|
| 1 | `request_id` | INTEGER PK | no | DB | Ordering and joins. Autoincrement, so it is also arrival order. |
| 2 | `run_id` | TEXT | no | API | P1's tuning-vs-reporting firewall. P5 quotes only from `eval_` runs. See the rule below. |
| 3 | `ts` | REAL | no | API | P1 — anything about rate, timing, or bursts. Unix epoch seconds, UTC, float. |
| 4 | `api_key_id` | TEXT | no | API | P1 — per-account behaviour (Stage 4), and which accounts filled the map together (Stage 6 attribution). `unknown` when auth failed. |
| 5 | `tier` | TEXT | no | API | P1 — is this volume paid for? An enterprise customer at high rate is not a suspect. `free` / `pro` / `enterprise`. |
| 6 | `ip` | TEXT | no | API | P4's office-behind-one-connection test. P1 derives the subnet from this in one line — we do not store a second chopped-up copy. |
| 7 | `status_code` | INTEGER | no | API | P5's dashboard. P3's Stage 9 robustness numbers. |
| 8 | `error_code` | TEXT | yes | API | P3 — Stage 9. Short machine tag: `bad_image`, `rate_limited`, `no_key`, `payload_too_large`. Null on success. |
| 9 | `latency_ms` | REAL | no | API | P5's dashboard. |
| 10 | `input_sha256` | TEXT | yes | API | P1 — junk padding traffic reuses images, and repeats show up here as identical hashes. This is how noise-based evasion gets caught. |
| 11 | `input_bytes` | INTEGER | yes | API | P1 — weak signal. |
| 12 | `embedding` | BLOB | yes | API | P1 — **the map.** 32 × float32, little-endian. See "the point, not the square" below. |
| 13 | `model_probs` | TEXT | yes | API | P1 — confidence, margin and uncertainty are all arithmetic on this. JSON list of 10 floats. **What the model believed.** |
| 14 | `returned_probs` | TEXT | yes | API | P2 — proving the theft. P5 — the defended-vs-undefended slide. JSON list of 10 floats. **What the customer actually received.** |
| 15 | `degradation_level` | INTEGER | no | API | P2, P5 — Stage 7. 0 = untouched, 1–3 = increasing damage. Always 0 until Stage 7 exists. |
| 16 | `suspicion_score` | REAL | yes | API | P5 — explains *why* the alarm fired at that moment. Null until Stage 7. |

---

## Two design decisions worth understanding

### The point, not the square

Column 12 stores *where in the model's knowledge a question landed* — a
point. It does **not** store which square of the map that point falls in.

That is deliberate, and it is the most important choice in this document.
P1 designs the map at Stage 6, on Day 2 afternoon. The moment they change
how squares are drawn, every square recorded before that change means
something different, and nothing tells you which rows came from which
design.

The point is a fact about what happened. The square is an *interpretation*
of that point, and interpretations belong to the detector, not the log. So
we log the point once, and P1 draws squares over it as many times and as
many ways as they like without regenerating a single request.

### Two probability lists, not one

Columns 13 and 14 are identical until Stage 7. That duplication is on
purpose.

From Stage 7 onward the API starts returning worse answers to suspicious
traffic. At that moment "what the model believed" and "what the customer
got" stop being the same thing — and that gap *is* the defence. Storing
both makes it a fact you can point at. Storing only one makes it a claim
you have to argue for.

Everything I originally had as separate summary columns (top label,
confidence, margin, entropy) is arithmetic on column 13. Deriving them
costs one line and cannot drift out of sync.

---

## The `run_id` rule — read this twice

`run_id` is how P1 keeps tuning numbers apart from reported numbers.

- Tuning runs: `run_id` begins `cal_` (e.g. `cal_seed1`)
- Reporting runs: `run_id` begins `eval_` (e.g. `eval_seed2`)

**P1 is the sole owner of which run is which.** Nothing tuned on a `cal_`
run may be quoted from that same run. P5 quotes from `eval_` runs only.

Judges explicitly penalise numbers reported on the data they were tuned
on, so this one text column protects a large part of the score.

---

## Table: `keys`

| Column | Type | Meaning |
|--------|------|---------|
| `api_key_id` | TEXT PK | e.g. `k_00042` |
| `secret` | TEXT | The value sent in the `X-API-Key` header |
| `tier` | TEXT | `free` / `pro` / `enterprise` |
| `owner` | TEXT | Ground-truth label: `casual`, `office_acme`, `attacker_pool`, … |
| `created_ts` | REAL | Unix epoch |

**`owner` is ground truth for scoring only. The detector must never read
it.** P1 uses it after the fact to check who was caught. If it ever leaks
into a feature, every number in the submission is worthless — the detector
would be reading the answer key.

---

## Rules that keep the log trustworthy

**Only the API writes rows.** Not the attacker, not the traffic generator.
Those send requests; the API records what arrived. The moment a second
program can write directly, the log stops being evidence of what happened
and becomes a mix of what happened and what someone claimed happened.

**Only `api/logstore.py` touches the database.** Everyone reads through
`read_df()`, which hands back a DataFrame with column 12 already decoded
into a numpy array and columns 13–14 already decoded into Python lists.
Nobody parses raw bytes by hand. If you need something that isn't here,
ask P3 rather than opening a second reader.

**Fake and real logs are siblings, not copies.** `api/fake_logs.py`
invents rows that satisfy this contract; the API records rows that satisfy
this contract. Neither is derived from the other, which is why P1 can swap
one for the other on Day 1 evening without changing a line.

---

## Open questions to settle at sign-off

1. **`latency_ms` (column 9)** — P5's dashboard is the only named reader,
   and a dashboard could live without it. It is free to record. Keep it
   as a nice-to-have, or cut it by our own rule?
2. **`input_bytes` (column 11)** — a weak signal at best. Same question.

I kept both because they cost nothing, but by the "must have a named
reader" filter above they are the two weakest entries in the table. The
team decides, not me.

---

## Deliberately absent

Recorded here so nobody re-proposes them later:

- **Browser / user-agent string** — no reader. Our own generators write
  it, so it would be a field describing our own code.
- **Endpoint name** — the same value on every row of this table.
- **A pre-chopped subnet column** — one line to derive from column 6.
- **An arrival counter** — column 1 already is one.
- **Cell / square id** — see "the point, not the square" above.

---

## Frozen-by sign-off

Sign only if you have found the columns your own stage needs.

| Person | Role | What you should have checked | Signed |
|--------|------|------------------------------|--------|
| P1 | Detection | Columns 3–6 for per-account signals, 10 and 12 for the map, 2 for the firewall | ☐ |
| P2 | Attack | Column 14 gives you exactly what your clone trains on; 15 lets you compare defended vs undefended | ☐ |
| P3 | Platform | You can actually produce every column at request time | ☐ |
| P4 | Traffic | Column 6 plus the `keys` table supports the office-on-one-connection test | ☐ |
| P5 | Evidence | Columns 2, 13, 14, 15, 16 support every chart you plan to draw | ☐ |

**Frozen on:** 09 September 2026