# B3 Temperature Normalization — ASCII Visualization

## Step 0 — Raw data `temps`, shape (14,)

```
Day:    0     1     2     3     4     5     6     7     8     9    10    11    12    13
      ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
      │  71 │  72 │  70 │  75 │  71 │  72 │  70 │  75 │  80 │  85 │  80 │  78 │  72 │  83 │
      └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
      │◄────────────── Week 0 ──────────────►│◄────────────── Week 1 ──────────────►│
```

---

## Step 1 — `einops.reduce(temps, "(h 7) -> h", "mean")` → `avg`, shape (2,)

Conceptually splits the flat (14,) into a (2, 7) grid, then takes the mean of each row:

```
        Week 0                          Week 1
      ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐
      │  71 │  72 │  70 │  75 │  71 │  72 │  70 │   mean → 71.571
      ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
      │  75 │  80 │  85 │  80 │  78 │  72 │  83 │   mean → 79.000
      └─────┴─────┴─────┴─────┴─────┴─────┴─────┘

avg = [ 71.571,  79.000 ]   shape (2,)
```

---

## Step 2 — `einops.reduce(temps, "(h 7) -> h", t.std)` → `std`, shape (2,)

Same grouping, but reduce with standard deviation instead of mean:

```
        Week 0                          Week 1
      ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐
      │  71 │  72 │  70 │  75 │  71 │  72 │  70 │   std  →  1.718
      ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
      │  75 │  80 │  85 │  80 │  78 │  72 │  83 │   std  →  4.472
      └─────┴─────┴─────┴─────┴─────┴─────┴─────┘

std = [  1.718,   4.472 ]   shape (2,)
```

---

## Step 3 — `einops.repeat(avg, "w -> (w 7)")` → shape (14,)

Each weekly scalar is tiled 7 times to match the original flat shape.
The `(w 7)` pattern means: for each of the w=2 values, emit 7 copies before moving on.

```
avg       [ 71.571,                    79.000 ]
              │                           │
              ▼  × 7                      ▼  × 7
avg_rep   [ 71.571, 71.571, 71.571, 71.571, 71.571, 71.571, 71.571,
             79.000,  79.000,  79.000,  79.000,  79.000,  79.000,  79.000 ]

          shape (14,)  — each element aligns with the day it belongs to
```

Same operation for `std`:

```
std       [  1.718,                     4.472 ]
              │                           │
              ▼  × 7                      ▼  × 7
std_rep   [  1.718,  1.718,  1.718,  1.718,  1.718,  1.718,  1.718,
             4.472,  4.472,  4.472,  4.472,  4.472,  4.472,  4.472 ]

          shape (14,)
```

---

## Step 4 — `(temps - avg_rep) / std_rep` element-wise → z-scores, shape (14,)

```
Day:       0       1       2       3       4       5       6
temps:    71      72      70      75      71      72      70
avg_rep:  71.571  71.571  71.571  71.571  71.571  71.571  71.571
          ──────  ──────  ──────  ──────  ──────  ──────  ──────
diff:     -0.571  +0.429  -1.571  +3.429  -0.571  +0.429  -1.571
std_rep:   1.718   1.718   1.718   1.718   1.718   1.718   1.718
          ──────  ──────  ──────  ──────  ──────  ──────  ──────
z:        -0.333  +0.249  -0.915  +1.995  -0.333  +0.249  -0.915

Day:       7       8       9      10      11      12      13
temps:    75      80      85      80      78      72      83
avg_rep:  79.000  79.000  79.000  79.000  79.000  79.000  79.000
          ──────  ──────  ──────  ──────  ──────  ──────  ──────
diff:     -4.000  +1.000  +6.000  +1.000  -1.000  -7.000  +4.000
std_rep:   4.472   4.472   4.472   4.472   4.472   4.472   4.472
          ──────  ──────  ──────  ──────  ──────  ──────  ──────
z:        -0.894  +0.224  +1.342  +0.224  -0.224  -1.565  +0.894
```

---

## Full data flow summary

```
temps  (14,) ──► reduce "(h 7)->h" mean ──► avg (2,) ──► repeat "w->(w 7)" ──► avg_rep (14,)
                                                                                       │
                                                                               temps - avg_rep
                                                                                       │
temps  (14,) ──► reduce "(h 7)->h" std  ──► std (2,) ──► repeat "w->(w 7)" ──► std_rep (14,)
                                                                                       │
                                                                                    ÷ std_rep
                                                                                       │
                                                                                  z  (14,)
```

### Why not just `temps - avg` directly?

```
temps   shape: (14,)    [71, 72, 70, 75, 71, 72, 70, 75, 80, 85, 80, 78, 72, 83]
avg     shape:  (2,)    [71.571, 79.000]

PyTorch aligns from the RIGHT:
   (14,)
    (2,)   ← size 14 ≠ size 2, and neither is 1  →  ERROR ✗

After repeat:
   (14,)
   (14,)   ← same shape  →  element-wise subtraction ✓
```
