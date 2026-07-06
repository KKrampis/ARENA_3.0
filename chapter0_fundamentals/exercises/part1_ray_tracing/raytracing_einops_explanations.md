# Ray Tracing and Einops: A Deep Dive into Batched Tensor Operations

## Preface: Why This Section Matters

Section 0.1 is where the abstract tensor manipulations from the prerequisites become load-bearing.  section 0.0 you learned what `einops.repeat` and `einops.reduce` do to named axes in isolation. Here you apply them to a concrete computational problem — ray tracing — where the *reason* for each reshape is not aesthetic but geometric: you need to test every ray against every segment, or every ray against every triangle, simultaneously and without Python loops. Einops is the tool that makes that possible in a readable, debuggable way.

This document walks through every einops operation in the section, explaining exactly what data is being manipulated, why the shape transformation is needed, what the alternative would look like, and how the operation connects to the surrounding linear algebra.

---

## Section Roadmap and Learning Objectives

The chapter is divided into three progressive sections. Each one introduces a new layer of complexity — both geometrically (from 1D to 3D) and computationally (from a single pair to a full mesh) — and each layer builds directly on the previous one. The learning objectives for each section are not isolated skills; they are steps in a single arc from "what is a ray?" to "render a 3D Pikachu."

---

### 1️⃣ Rays & Segments

**What you build:** Functions to create batches of rays and test whether a single ray intersects a single 2D line segment.

**Learning objectives:**

> - Learn how to create PyTorch tensors in a variety of ways
> - Understand how to parametrize lines and rays in 2D
> - Learn about type annotations and linear operations in PyTorch

**What these objectives actually mean in practice:**

*Creating PyTorch tensors in a variety of ways* — the exercises introduce `t.zeros` (allocate and fill with zeros), `t.linspace` (evenly spaced values), the `out=` keyword argument (write values into an existing tensor's memory rather than allocating a new one), and `t.stack` (combine existing tensors into a new tensor along a new axis). Each of these is a different creation or assembly pattern, and recognising which tool fits which task is a foundational PyTorch skill that every subsequent exercise requires.

*Parametrising lines and rays in 2D* — a ray is not a single point but a continuous set of points described by the equation `P(u) = O + u * D`. A line segment similarly has a parametric form `P(v) = L1 + v * (L2 - L1)`. The conceptual leap here is moving from "a ray is a thing you shoot" to "a ray is a mathematical object with an algebraic description." Once both the ray and the segment are expressed as parametric equations, finding their intersection reduces to solving for `u` and `v` in a linear system — a purely algebraic problem that `torch.linalg.solve` handles exactly.

*Type annotations and linear operations* — `Float[Tensor, "points dims"]` is a type annotation that says: this tensor has floating-point values, and its shape has two named axes called `points` and `dims`. These annotations do nothing at runtime but serve as documentation that makes axis intent explicit — you can read a function signature and know immediately what shape to expect. The linear operation at the core of this section is `torch.linalg.solve(A, b)`, which solves the system `Ax = b` for `x` using LU decomposition. Understanding when this can fail (singular matrix, i.e. parallel ray and segment) and how to handle it (catch the `RuntimeError`) is a practical lesson in numerical linear algebra.

**Why it sets up everything else:** The single-pair function `intersect_ray_1d` is the mathematical template. Every function in sections 2 and 3 is the same computation extended over more pairs simultaneously. If you understand why the 2×2 matrix has `D` as its first column and `L1-L2` as its second column — and why the right-hand side is `L1-O` — then the 3×3 triangle system in section 3 will feel like a natural extension rather than a new concept.

---

### 2️⃣ Batched Operations

**What you build:** A vectorised version of the ray-segment intersection test that checks all `NR` rays against all `NS` segments simultaneously, plus a function that generates a 2D grid of rays for rendering a 2D screen.

**Learning objectives:**

> - Learn about some important concepts related to batched operations, e.g. broadcasting and logical reductions
> - Understand and use the einops library
> - Apply this knowledge to create & work with a batch of rays

**What these objectives actually mean in practice:**

*Batched operations and broadcasting* — the bottleneck in naive ray tracing is the double Python loop: for each of NR rays, test each of NS segments. The batched approach replaces the loops with a single tensor computation over the `(NR, NS)` grid of pairs. Broadcasting is the mechanism that makes arithmetic between tensors of different shapes work — when one tensor is shape `(NR, NS, 2)` and another is shape `(2,)`, PyTorch aligns them from the right and implicitly repeats the smaller tensor. But broadcasting alone cannot create the `(NR, NS, 2, 2)` batch of matrices needed here, because the two input tensors have incompatible leading shapes. This is where `einops.repeat` does the heavy lifting.

*Logical reductions* — `torch.any(dim=-1)` is the final step: after computing a `(NR, NS)` boolean matrix of individual intersection tests, you need to reduce it to a `(NR,)` vector of "did this ray hit anything?" answers. The `.any(dim=-1)` call collapses the `NS` segment axis by asking "is at least one entry True?" for each row. This pattern — expand to the full grid, compute, then reduce — is the core computational template for batched geometric tests and is reused with `.min` in section 3.

*The einops library in a batched context* — in section 0.0, `einops.repeat` was used to tile images. Here it creates the Cartesian product needed for batched intersection testing: broadcasting rays across all segments with `"nrays p d -> nrays nsegments p d"` and broadcasting segments across all rays with `"nsegments p d -> nrays nsegments p d"`. The string notation makes the intent geometric: you are inserting a new axis that represents "the other thing being paired with." The `make_rays_2d` function also uses `einops.repeat` with the `"y -> (y z)"` pattern to tile a 1D y-grid into a 2D pixel grid — the same axis-merging idea as the `(b h)` pattern in the image exercises.

*Applying this to a batch of rays* — `make_rays_2d` generalises `make_rays_1d` to two spatial dimensions. The camera now shoots a grid of `NY × NZ` rays, one per pixel of a 2D screen. The challenge is that a 2D pixel grid must be stored as a 1D ray list of length `NY * NZ`, so you need to "unroll" the grid into a flat sequence in a consistent order. The einops `(y z)` merge achieves this: y varies slowly (one block per row), z varies fast (cycles through all column positions for each row). This unrolling convention is then respected by reshaping the output back to `(NY, NZ)` for display.

**Why it sets up everything else:** The singular-matrix handling pattern introduced here — detect with `det.abs() < 1e-8`, replace with `t.eye`, solve, then mask out — is copied verbatim into `raytrace_triangle` and `raytrace_mesh` in section 3. The only change is the matrix size (2×2 → 3×3) and the number of batch axes. If you understand why the identity-matrix replacement trick works (it gives a safe solve result that is then discarded by the mask), the section 3 code is immediately readable.

---

### 3️⃣ Triangles

**What you build:** A renderer for a full 3D mesh — specifically a Pikachu model made of 412 triangles — that takes a 2D grid of rays and returns a 2D image where each pixel's brightness encodes the depth of the nearest triangle intersection.

**Learning objectives:**

> - Understand how to parametrize triangles in 2D and 3D, and solve for their intersection with rays
> - Put everything together, to render your mesh as a 2D image

**What these objectives actually mean in practice:**

*Parametrising triangles* — a triangle in 3D is the set of points reachable as convex combinations of its three vertices `A`, `B`, `C`. In barycentric form, any point inside the triangle is `P = A + u*(B-A) + v*(C-A)` where `u >= 0`, `v >= 0`, and `u + v <= 1`. These three conditions on `(u, v)` are the triangle's "inside" test — they replace the single condition `0 <= v <= 1` from the segment case. Setting `P = O + sD` (the ray equation) and rearranging gives a 3×3 linear system with unknowns `(s, u, v)`, where `s` is the distance along the ray to the triangle's plane, and `(u, v)` are the barycentric coordinates of the hit point.

*Solving for the intersection* — the same `torch.linalg.solve` machinery from sections 1 and 2 applies, but now on a 3×3 system: `[-D | B-A | C-A] * [s, u, v]^T = O - A`. The three column vectors in the matrix are the negated ray direction and the two edge vectors of the triangle emanating from vertex `A`. The columns are constructed with `t.stack([-D, B-A, C-A], dim=-1)`, exactly the same `dim=-1` pattern as in section 1. The validity conditions add one more check — `u + v <= 1` — on top of the earlier `u >= 0`, `v >= 0`, `s >= 0`.

*Putting it all together: rendering a mesh* — `raytrace_mesh` is the culmination. The einops pattern from section 2 is extended to two batch axes simultaneously: `einops.repeat(triangles, "NT pts dims -> pts NR NT dims", NR=NR)` and `einops.repeat(rays, "NR pts dims -> pts NR NT dims", NT=NT)` create the full `(NR, NT)` grid of problems. The solve runs over the `(NR, NT, 3, 3)` batch. The result is a `(NR, NT)` matrix of distances `s`, where entries that failed the validity test are set to `float("inf")`. A final `einops.reduce(s, "NR NT -> NR", "min")` collapses the triangle axis, giving the nearest intersection distance for each ray. Reshaping to `(NY, NZ)` produces the depth image.

**Why this is the end goal:** The whole chapter exists to build up to this renderer. The single-pair functions in section 1 established the math. The batching techniques in section 2 established the vectorisation pattern. Section 3 combines both: the 3×3 linear system from triangle geometry slotted into the batching framework developed for segments, scaled up to a full mesh. The resulting code — roughly 20 lines — renders a 3D model without a single Python loop, entirely through batched tensor operations.

---

**The progression in one sentence per section:**

| Section                | Geometry                            | Computation                   | Einops role                                      |
| ---------------------- | ----------------------------------- | ----------------------------- | ------------------------------------------------ |
| 1 — Rays & Segments    | 2D line intersection                | Single 2×2 solve              | None (single pair, no batching needed)           |
| 2 — Batched Operations | Same 2D intersection, many pairs    | `(NR, NS)` grid of 2×2 solves | `repeat` to build the grid; `reduce` via `.any`  |
| 3 — Triangles          | 3D triangle intersection, full mesh | `(NR, NT)` grid of 3×3 solves | `repeat` for both axes; `reduce "min"` for depth |

---

---

## The Data Model: Understanding the Tensors Before Any Einops

Before examining any einops call, it is worth being precise about what every tensor in this section actually represents, because the shapes are not arbitrary — they encode geometric meaning.

### Rays

A **ray** is defined by two points in 3D space: an **origin** `O` (where the ray starts) and a **direction** `D` (which, added to the origin, gives a second point on the ray's path). Together they parametrise the infinite half-line:

```
P(t) = O + t * D,    t >= 0
```

The parameter `t` controls how far along the ray you travel. When `t = 0` you are at the origin. When `t = 1` you are at the point `O + D`. Negative values of `t` would represent points *behind* the camera, which is why the valid intersection condition requires `t >= 0` (equivalently written as `u >= 0` or `s >= 0` in the code). The direction vector `D` does not need to be a unit vector — it is simply the displacement from the origin to a reference point on the ray's path, and `t` scales it.

#### A single ray as a tensor: shape `(2, 3)`

In code, a single ray is stored as a tensor of shape `(2, 3)`:

```
ray = [[Ox, Oy, Oz],   ← row 0: the origin point O
       [Dx, Dy, Dz]]   ← row 1: the direction point D
```

The first axis (size 2) is the "which point" axis — row 0 is always the origin, row 1 is always the direction. The second axis (size 3) is the spatial coordinate — column 0 is x, column 1 is y, column 2 is z. So `ray[0, 1]` is `Oy` (the y-coordinate of the origin) and `ray[1, 0]` is `Dx` (the x-component of the direction).

To extract `O` and `D` from a single ray you unpack along the first axis:

```python
O, D = ray        # O = ray[0], shape (3,)
                  # D = ray[1], shape (3,)
```

#### A batch of NR rays: shape `(NR, 2, 3)`

When the camera shoots `NR` rays simultaneously — one per pixel — they are stacked into a single 3D tensor:

```
rays[r, p, d]

  r : ray index       0 … NR-1      which of the NR rays
  p : point index     0 or 1        0 = origin O, 1 = direction D
  d : dimension       0, 1, or 2    x, y, or z coordinate
```

Concretely for `NR = 4`:

```
rays = [
  [[O0x, O0y, O0z], [D0x, D0y, D0z]],   ← ray 0: origin and direction
  [[O1x, O1y, O1z], [D1x, D1y, D1z]],   ← ray 1
  [[O2x, O2y, O2z], [D2x, D2y, D2z]],   ← ray 2
  [[O3x, O3y, O3z], [D3x, D3y, D3z]],   ← ray 3
]
shape: (4, 2, 3)
```

Each "slice" `rays[r]` is a `(2, 3)` sub-tensor holding the single ray `r`. Slicing along the point axis gives all origins or all directions at once:

```python
origins    = rays[:, 0, :]   # shape (NR, 3) — O for every ray
directions = rays[:, 1, :]   # shape (NR, 3) — D for every ray
```

Or equivalently with `unbind`:

```python
O, D = rays.unbind(dim=1)    # O shape (NR, 3), D shape (NR, 3)
```

This layout — ray index outermost, point index in the middle, spatial dimension innermost — is the layout used throughout the section. It is a design choice: it keeps "all data for ray r" contiguous in memory (`rays[r]`), and it makes unbinding along `dim=1` the natural way to separate origins from directions.

#### `make_rays_1d` — constructing the batch from scratch

The function that creates the initial ray batch for 1D rendering illustrates how all three axes are populated:

```python
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    rays = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    #               ──────────  ─  ─
    #               NR rays     2  3D coords
    #                           points per ray
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    rays[:, 1, 0] = 1
    return rays
```

Step by step:

1. `t.zeros((num_pixels, 2, 3))` — allocates the full `(NR, 2, 3)` tensor filled with zeros. At this point every origin is `[0, 0, 0]` and every direction is `[0, 0, 0]`.

2. `rays[:, 1, 0] = 1` — sets the x-component of every direction to 1. Index breakdown: `[:, 1, 0]` means "all rays (`:`) → direction point (`1`) → x-coordinate (`0`)". After this line, `D = [1, 0, 0]` for every ray — all rays point in the positive x direction.

3. `t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])` — fills the y-component of every direction with evenly spaced values from `-y_limit` to `+y_limit`. The `out=` keyword writes directly into the existing tensor's memory rather than creating a new one. Index breakdown: `[:, 1, 1]` means "all rays → direction point → y-coordinate". After this line, ray `r` has `Dy = -y_limit + r * (2*y_limit / (num_pixels-1))`.

The result for `num_pixels=5, y_limit=1.0`:

```
rays = [
  [[0, 0, 0], [1, -1.00, 0]],   ← ray 0: points toward upper edge
  [[0, 0, 0], [1, -0.50, 0]],   ← ray 1
  [[0, 0, 0], [1,  0.00, 0]],   ← ray 2: points straight ahead
  [[0, 0, 0], [1,  0.50, 0]],   ← ray 3
  [[0, 0, 0], [1,  1.00, 0]],   ← ray 4: points toward lower edge
]
shape: (5, 2, 3)

Memory layout (all values in row-major order):
  axis 0 (ray):       0    0    0    0    0    0    1    1    1    1    1    1  ...
  axis 1 (point):     0    0    0    1    1    1    0    0    0    1    1    1  ...
  axis 2 (dim):       x    y    z    x    y    z    x    y    z    x    y    z  ...
  values:             0    0    0    1   -1    0    0    0    0    1  -.5    0  ...
                      ←─ ray 0 origin ─→  ←─ ray 0 direction ─→  ←─ ray 1 origin ─→  ...
```

All origins remain `[0, 0, 0]` (the camera is at the origin). Only the direction's y-component varies — this is what makes each ray point at a different pixel on the 1D screen at `x = 1`.

#### The axis layout is a contract, not a coincidence

The `(NR, 2, 3)` shape is maintained consistently across the entire section. Every function that accepts `rays` as input can rely on `rays[:, 0]` being origins and `rays[:, 1]` being directions. Every function that creates rays uses `t.zeros((n, 2, 3))` as its starting point. This consistency is what makes `O, D = rays.unbind(dim=1)` work everywhere — it is the agreed-upon convention for how rays are encoded in memory.

### Line Segments (Section 2)

A **line segment** in 2D is defined by two endpoints `L1` and `L2`. In code a single segment is stored as `(2, 3)` (same shape as a ray, with z=0 ignored). A batch of `NS` segments is `(NS, 2, 3)`.

### Triangles (Section 3)

A **triangle** in 3D is defined by three vertices `A`, `B`, `C`, each a point in `(x, y, z)`. A single triangle is stored as `(3, 3)`:

```
triangle = [[Ax, Ay, Az],   ← vertex A (row 0)
            [Bx, By, Bz],   ← vertex B (row 1)
            [Cx, Cy, Cz]]   ← vertex C (row 2)
```

A mesh of `NT` triangles is `(NT, 3, 3)`.

### The Core Challenge

The computational heart of ray tracing is answering: **does ray `r` intersect object `o`?** For a scene with `NR` rays and `NS` segments (or `NT` triangles), the brute-force answer requires testing every `(ray, object)` pair. That is `NR × NS` (or `NR × NT`) tests. To do this without Python loops — and therefore at GPU speed — you need both tensors expanded to the same shape, so that a single vectorised linear solve covers the entire matrix of pairs at once.

Einops is what performs that expansion in a geometrically interpretable, axis-named way.

---

## Section 1: Single Ray, Single Segment — The Baseline (No Einops)

Before the batched code, the single-pair function `intersect_ray_1d` establishes the mathematical skeleton that all later functions will vectorise. It is worth understanding this derivation completely, because every subsequent function — batched over segments, batched over triangles, extended to 3D — is a direct generalisation of the same steps.

### Step 1: Two parametric line equations

In the 2D case (z is dropped throughout section 1), both the ray and the segment are described as parametric lines — equations that trace out a set of points as a scalar parameter varies.

**The ray** starts at origin `O = (Ox, Oy)` and travels in direction `D = (Dx, Dy)`. Any point on the ray can be written as:

```
P_ray(u) = O + u * D
         = (Ox + u*Dx,  Oy + u*Dy)
```

The parameter `u` is a scalar. When `u = 0` you are at the origin. When `u = 1` you are at `O + D`. Only `u >= 0` is valid — negative `u` would trace the ray backwards behind the camera.

**The segment** runs from endpoint `L1 = (L1x, L1y)` to endpoint `L2 = (L2x, L2y)`. Its parametric form uses a convex combination:

```
P_seg(v) = L1 + v * (L2 - L1)
         = (L1x + v*(L2x-L1x),  L1y + v*(L2y-L1y))h
```

When `v = 0` you are at `L1`. When `v = 1` you are at `L2`. The parameter `v` must lie in `[0, 1]` for the point to be on the segment rather than on its infinite extension beyond the endpoints.

### Step 2: Setting the equations equal — what "intersection" means

An intersection point is a single point in 2D space that lies on both the ray and the segment simultaneously. That means there exist values of `u` and `v` such that:

```
P_ray(u) = P_seg(v)
O + u * D = L1 + v * (L2 - L1)
```

This is one vector equation in 2D, which expands to two scalar equations (one per coordinate):

```
Ox + u * Dx = L1x + v * (L2x - L1x)   ← x-equation
Oy + u * Dy = L1y + v * (L2y - L1y)   ← y-equation
```

The unknowns are `u` and `v`. Everything else (`Ox, Oy, Dx, Dy, L1x, L1y, L2x, L2y`) is known data from the input tensors.

### Step 3: Rearranging into a standard linear system Ax = b

Move all `u` and `v` terms to the left, all constants to the right:

```
u * Dx - v * (L2x - L1x) = L1x - Ox
u * Dy - v * (L2y - L1y) = L1y - Oy
```

Note the sign: `v * (L2 - L1)` on the right becomes `-v * (L2 - L1)` on the left, which is the same as `+v * (L1 - L2)`. This explains why the matrix column is `L1 - L2`, not `L2 - L1`. Written in matrix form `A * x = b`:

```
┌                        ┐ ┌   ┐   ┌           ┐
│  Dx    (L1x - L2x)     │ │ u │   │ L1x - Ox  │
│  Dy    (L1y - L2y)     │ │ v │ = │ L1y - Oy  │
└                        ┘ └   ┘   └           ┘

     A (2×2 matrix)          x        b (right-hand side)
```

The matrix `A` has:

- **First column** = the direction vector `D = [Dx, Dy]`
- **Second column** = the vector `L1 - L2 = [L1x-L2x, L1y-L2y]`

The right-hand side `b` is `L1 - O = [L1x-Ox, L1y-Oy]`.

### Step 4: Geometric interpretation of the solution

`t.linalg.solve(A, b)` finds the unique `(u, v)` satisfying `A * [u, v]^T = b`, if the system has a unique solution.

- **`u`** is the signed scale factor along the ray from the origin to the intersection point, in units of `|D|`. A ray hits an intersection in front of the camera when `u >= 0`.
- **`v`** is the fractional position along the segment from `L1` to `L2`. The hit is within the segment when `0 <= v <= 1`. If `v < 0` the intersection is on the extension beyond `L1`. If `v > 1` it is beyond `L2`.

Geometrically:

```
            u=0          u>0 (valid hit)
             │
  Camera ────O──────────────►────── direction D ──────────────►
                            ↑
                     intersection point
                            │
             L1 ────────────┼──────────────── L2
             v=0     0<v<1 (valid)          v=1
```

If the ray and segment are parallel, the system has no unique solution — the matrix `A` is singular (determinant = 0) and `t.linalg.solve` raises a `RuntimeError`, which the function catches and converts to `False`.

### Step 5: Why the system can be singular — the parallel case

Two lines are parallel when their direction vectors are proportional. The determinant of `A` is:

```
det(A) = Dx * (L1y - L2y) - Dy * (L1x - L2x)
```

This is the 2D cross product of `D` and `(L2 - L1)`. It equals zero when the two vectors point in the same (or exactly opposite) direction — i.e., when the ray and segment run parallel to each other.

```
Case 1: non-parallel (generic)
  D        = [1, 0]    → ray points right
  L2 - L1  = [0, 1]   → segment points up
  det(A)   = 1*1 - 0*0 = 1   → non-zero, unique solution exists ✓

Case 2: parallel (singular)
  D        = [1, 0]    → ray points right
  L2 - L1  = [2, 0]   → segment also points right
  det(A)   = 1*0 - 0*2 = 0   → zero, no unique intersection ✗
```

Parallel rays and segments either never intersect (if they are offset) or overlap everywhere (if they are collinear). In both cases there is no single intersection point, so returning `False` is correct.

### Step 6: Concrete worked example

Take a ray from the origin pointing right-and-up, and a vertical segment:

```
O  = (0, 0)       camera at origin
D  = (2, 1)       ray direction: 2 units right, 1 unit up
L1 = (3, -1)      segment bottom endpoint
L2 = (3, +2)      segment top endpoint
```

Build the matrix and right-hand side:

```
A = [ Dx       L1x-L2x ]   =  [ 2    3-3  ]   =  [ 2   0 ]
    [ Dy       L1y-L2y ]      [ 1   -1-2  ]      [ 1  -3 ]

b = [ L1x - Ox ]   =  [ 3 - 0 ]   =  [  3 ]
    [ L1y - Oy ]      [-1 - 0 ]      [ -1 ]
```

Solve `A * [u, v]^T = b` by row reduction:

```
Row 0:  2u + 0v =  3   →  u = 3/2 = 1.5
Row 1:  1u - 3v = -1   →  substitute u=1.5:  1.5 - 3v = -1
                        →  3v = 2.5
                        →  v = 5/6 ≈ 0.833
```

Check the validity conditions:

- `u = 1.5 >= 0` ✓  (the intersection is ahead of the camera)
- `v = 0.833 ∈ [0, 1]` ✓  (the intersection is within the segment's endpoints)
- Result: **True** — ray and segment intersect.

Verify geometrically:

```
Intersection via ray:     O + u*D = (0,0) + 1.5*(2,1) = (3.0, 1.5)
Intersection via segment: L1 + v*(L2-L1) = (3,-1) + 0.833*(0,3) = (3.0, 1.5)  ✓
```

Both parametric equations give the same point `(3.0, 1.5)`.

### Step 7: The tensor operations in the code, explained line by line

```python
def intersect_ray_1d(ray, segment) -> bool:
    # Drop the z coordinate — working in 2D only.
    # ray[:, :2] keeps columns 0 and 1 (x and y), drops column 2 (z).
    ray     = ray[:, :2]        # (2, 3) → (2, 2)
    segment = segment[:, :2]    # (2, 3) → (2, 2)

    # Unpack rows of each (2, 2) tensor along axis 0.
    # ray[0] = [Ox, Oy], ray[1] = [Dx, Dy]
    O, D     = ray              # O shape (2,), D shape (2,)
    L_1, L_2 = segment         # L_1 shape (2,), L_2 shape (2,)

    # Build the 2×2 matrix A by stacking two column vectors.
    # D = [Dx, Dy] and L_1 - L_2 = [L1x-L2x, L1y-L2y] are both shape (2,).
    # t.stack([...], dim=-1) treats each input as a column and places them side by side.
    # dim=-1 means "stack along the last (new) axis" → each vector becomes a column.
    mat = t.stack([D, L_1 - L_2], dim=-1)   # shape (2, 2)
    #   mat[0, 0] = Dx      mat[0, 1] = L1x - L2x   ← first row
    #   mat[1, 0] = Dy      mat[1, 1] = L1y - L2y   ← second row
    #              col 0              col 1

    # Build the right-hand side vector b = L1 - O.
    vec = L_1 - O                            # shape (2,): [L1x-Ox, L1y-Oy]

    # Solve A * [u, v]^T = b.
    # If A is singular (det ≈ 0, i.e. ray parallel to segment), raises RuntimeError.
    try:
        sol = t.linalg.solve(mat, vec)       # shape (2,): [u, v]
    except RuntimeError:
        return False

    # Extract Python floats from 0-dim tensors via .item().
    # sol[0] is the tensor at index 0 (a scalar tensor), .item() unwraps to float.
    u = sol[0].item()
    v = sol[1].item()

    # Intersection is valid when u >= 0 (in front of camera) and v in [0,1] (on segment).
    return (u >= 0.0) and (v >= 0.0) and (v <= 1.0)
```

#### Why `dim=-1` in `t.stack`?

This is a subtle but important detail. `t.stack` creates a **new axis** and places the input tensors along it. `dim=-1` means the new axis goes at the end — after all existing axes. Since `D` and `L_1 - L_2` are both 1D tensors of shape `(2,)`, the existing axis is at position 0 (size 2), so `dim=-1` places the new axis at position 1:

```
D         = [Dx, Dy]               shape (2,)
L_1 - L_2 = [L1x-L2x, L1y-L2y]   shape (2,)

t.stack([D, L_1-L_2], dim=0):    → [[Dx, Dy], [L1x-L2x, L1y-L2y]]   shape (2, 2) — stacked as ROWS
t.stack([D, L_1-L_2], dim=-1):   → [[Dx, L1x-L2x], [Dy, L1y-L2y]]   shape (2, 2) — stacked as COLUMNS
```

The matrix equation requires `D` and `L1-L2` as **columns**, not rows — that is, the coefficient of `u` in the x-equation is `Dx` (first element of column 0) and the coefficient of `u` in the y-equation is `Dy` (second element of column 0). Using `dim=0` would give the transpose of the correct matrix, producing wrong results. This same `dim=-1` pattern scales directly to the 3×3 case in section 3, where three column vectors are stacked: `t.stack([-D, B-A, C-A], dim=-1)`.

No einops appears in this function. All tensors are at most 2D, plain Python unpacking (`O, D = ray`) handles the axis splitting, and a `try/except` handles the singular case. Einops enters precisely in the next section, when this same computation must be repeated in parallel over all `(NR, NS)` pairs simultaneously.

---

## Section 1 Code Walkthrough — All Segments

The notebook's Section 1 contains four distinct code blocks beyond the functions already covered above. Each one introduces a technique or pattern that reappears throughout the chapter.

---

#### Code Segment 1: Allocating the ray tensor with `t.zeros` and the `out=` keyword

```python
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    rays = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    rays[:, 1, 0] = 1
    return rays
```

**`t.zeros((num_pixels, 2, 3), dtype=t.float32)`**

This allocates a tensor of the exact shape needed — `(NR, 2, 3)` — and fills every element with zero. The `dtype=t.float32` argument forces 32-bit floating point, which is the standard precision for graphics and deep learning (64-bit would work but uses twice the memory with no benefit here). At this point:

```
rays[:, 0, :] = [[0, 0, 0], ...]   ← all origins are the camera at (0,0,0)
rays[:, 1, :] = [[0, 0, 0], ...]   ← all directions are zero (not yet valid)
```

**`rays[:, 1, 0] = 1`**

This is a broadcasting assignment into a slice. The index `[:, 1, 0]` selects:

- `:` — all rays (axis 0, all NR of them)
- `1` — the direction row (axis 1, point index 1)
- `0` — the x-coordinate (axis 2, dimension 0)

It sets the x-component of every direction to 1 simultaneously, without a loop. After this line all directions are `[1, 0, 0]` — every ray points in the positive x direction.

**`t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])`**

`t.linspace(start, end, steps)` normally returns a new 1D tensor of `steps` evenly spaced values from `start` to `end` inclusive. The `out=` keyword argument changes that behaviour: instead of allocating a new tensor, it writes the result directly into the memory location provided. Here `rays[:, 1, 1]` is a view of shape `(NR,)` into the existing `rays` tensor — specifically the y-coordinate of every direction. Writing into this view modifies the underlying `rays` data in place, with zero extra allocation. This is a memory-efficient pattern that becomes important when tensors are large.

The result is that ray `r` gets direction y-component `Dy = -y_limit + r * (2 * y_limit / (num_pixels - 1))`. Combined with `Dx = 1` and `Dz = 0`, ray `r` points toward screen position `y = Dy` at `x = 1`.

---

#### Code Segment 2: Visualising the rays with `render_lines_with_plotly`

```python
rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)
```

`render_lines_with_plotly` is a utility function defined in `utils.py`. It takes the `(NR, 2, 3)` rays tensor and draws each ray as a line in a 3D Plotly figure, with one endpoint at the origin `rays[r, 0]` and the other at `rays[r, 0] + rays[r, 1]` (i.e. `O + D`). This is a sanity-check step: before doing any intersection math, you verify visually that the rays fan out as expected — all starting at `(0,0,0)` and spreading across a range of y-values at `x=1`. In practice, anytime you write a tensor-building function like `make_rays_1d`, plotting the result immediately is the fastest way to catch shape or indexing bugs before they propagate into more complex functions.

The `render_lines_with_plotly` function accepts an optional second argument for additional line segments to overlay — this is used later in the exercises to draw both the rays and the object (segment or triangle outline) in the same figure, making it easy to visually confirm whether an intersection should exist.

---

#### Code Segment 3: The interactive widget

```python
fig: go.FigureWidget = setup_widget_fig_ray()
display(fig)

@interact(v=(0.0, 6.0, 0.01), seed=(0, 10, 1))
def update(v=0.5, seed=0):
    ...
```

This block is a Jupyter widget that lets you drag a slider to vary the parameter `v` and watch the intersection geometry update in real time. The `@interact` decorator from the `ipywidgets` library wraps the `update` function and automatically creates UI sliders for each parameter — `v` ranging from 0.0 to 6.0 in steps of 0.01, and `seed` from 0 to 10 in integer steps. Each time you move a slider, `update` is called with the new values, it recomputes the ray and segment geometry, and it mutates the `fig` object to redraw.

The purpose of this widget is to build geometric intuition for what `u` and `v` mean before writing any code. By dragging the `v` slider you can see the intersection point slide along the segment, and you can observe what happens when `v` goes outside `[0, 1]` — the intersection point exists mathematically but is off the end of the segment, so the function should return `False`. The `seed` slider generates different random segments so you can test the intuition across multiple geometric configurations.

---

#### Code Segment 4: The `intersect_ray_1d` function and type annotations

```python
def intersect_ray_1d(
    ray: Float[Tensor, "points dims"],
    segment: Float[Tensor, "points dims"]
) -> bool:
    ...

tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
```

**Type annotations with `Float[Tensor, "points dims"]`**

The annotations `Float[Tensor, "points dims"]` come from the `jaxtyping` library (imported as `from jaxtyping import Float`). They are not enforced at runtime by default — Python ignores them during execution. Their value is purely documentary: anyone reading the function signature immediately knows that `ray` is a floating-point tensor with two named axes called `points` (size 2: origin and direction) and `dims` (size 3: x, y, z). Without this annotation, the parameter is just called `ray: Tensor` and you have to read the function body to figure out the expected shape. With the annotation, the shape contract is part of the interface.

More specifically, `Float[Tensor, "points dims"]` says:

- `Float` — dtype is floating point (float16, float32, or float64)
- `Tensor` — it is a PyTorch tensor specifically
- `"points dims"` — it has two axes, named `points` and `dims` (names are chosen to be semantically meaningful, not enforced by size)

If you install the `beartype` library alongside `jaxtyping`, the annotations *are* enforced at runtime — a tensor of the wrong dtype or number of dimensions raises a `TypeError` immediately, making shape bugs easy to catch. This is optional but recommended when debugging.

**`tests.test_intersect_ray_1d(intersect_ray_1d)`**

The test functions in `tests.py` construct specific ray-segment pairs where the correct answer is known analytically, call your implementation, and `assert` that it returns the expected value. Running them immediately after implementing a function is the standard workflow throughout ARENA — you write the function body, run the tests, and either see an `AssertionError` (with a message pointing to what went wrong) or silence (indicating the tests passed).

`test_intersect_ray_1d_special_case` specifically tests the parallel-ray edge case — the singular matrix scenario — to make sure the `try/except RuntimeError` branch is actually triggered and returns `False` correctly, rather than crashing or silently returning a wrong answer.

---

#### The Setup Code (top of notebook)

```python
import os, sys
from pathlib import Path
import einops
import torch as t
from torch import Tensor
from jaxtyping import Float, Bool
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import interact
```

These imports establish the toolkit for the entire section. Each library has a specific role:

| Import                           | Role in Section 1                                                                         |
| -------------------------------- | ----------------------------------------------------------------------------------------- |
| `torch as t`                     | All tensor creation and arithmetic — `t.zeros`, `t.linspace`, `t.stack`, `t.linalg.solve` |
| `einops`                         | Not used in section 1; enters in section 2 for batching                                   |
| `Float, Bool` from `jaxtyping`   | Shape-annotated type hints on function signatures                                         |
| `Tensor` from `torch`            | Base type used in annotations                                                             |
| `plotly.express / graph_objects` | 3D visualisation of rays and segments                                                     |
| `interact` from `ipywidgets`     | Slider widgets for interactive geometry exploration                                       |
| `Path` from `pathlib`            | File-system paths for loading the Pikachu mesh in section 3                               |

The `section_dir` path variable set up in the setup cell points to the directory containing `pikachu.pt` — the mesh file used in section 3. The setup code also adds the repo root to `sys.path` so that `import tests` and `import utils` resolve correctly regardless of which directory the notebook is opened from.

---

## Section 2: Batched Ray-Segment Intersection — Where Einops First Appears

### The Problem with Loops

Suppose you have `NR = 200` rays and `NS = 50` segments. The naive approach:

```python
results = []
for ray in rays:
    hit = False
    for segment in segments:
        if intersect_ray_1d(ray, segment):
            hit = True
    results.append(hit)
```

This calls `intersect_ray_1d` 10,000 times, each time doing a Python function call, a PyTorch solve, and a boolean check. On CPU this is slow; on GPU it is catastrophic because each call launches a separate kernel and destroys all parallelism.

The batched approach solves *all* 10,000 pairs at once with a single call to `t.linalg.solve` operating on a `(NR, NS, 2, 2)` batch of matrices. Getting to that shape requires einops.

---

### `einops.repeat` #1 — Broadcasting Rays Across Segments

**Location:** `intersect_rays_1d`

```python
rays = einops.repeat(rays, "nrays p d -> nrays nsegments p d", nsegments=NS)
```

#### What the input tensor looks like

After slicing to xy, `rays` has shape `(NR, 2, 2)`:

```
axis 0: nrays   — which ray (0 … NR-1)
axis 1: p       — point index (0=origin O, 1=direction D)
axis 2: d       — spatial dimension (x, y)
```

Concretely for NR=3:

```
rays[0] = [[O0x, O0y], [D0x, D0y]]   ← ray 0
rays[1] = [[O1x, O1y], [D1x, D1y]]   ← ray 1
rays[2] = [[O2x, O2y], [D2x, D2y]]   ← ray 2
```

#### What the output tensor looks like

After `repeat`, shape is `(NR, NS, 2, 2)`:

```
axis 0: nrays      — which ray
axis 1: nsegments  — which segment (NEW axis, inserted by repeat)
axis 2: p          — point index
axis 3: d          — spatial dimension
```

For NR=3, NS=2, the layout is:

```
result[0, 0] = rays[0]   ray 0 paired with segment 0
result[0, 1] = rays[0]   ray 0 paired with segment 1
result[1, 0] = rays[1]   ray 1 paired with segment 0
result[1, 1] = rays[1]   ray 1 paired with segment 1
result[2, 0] = rays[2]   ray 2 paired with segment 0
result[2, 1] = rays[2]   ray 2 paired with segment 1
```

Each individual ray is copied `NS` times along the new `nsegments` axis. The copy is exact: `result[r, :, :, :] == rays[r]` for every `r`.

#### Why this is needed

The linear system for pair `(r, s)` is:

```
mat[r, s] = [D[r] | (L1[s] - L2[s])]     shape (2, 2)
vec[r, s] = L1[s] - O[r]                  shape (2,)
```

The matrix has a column from ray `r` and a column from segment `s`. To construct `mat[r, s]` for all `(r, s)` simultaneously, you need `D[r]` to be available at every `(r, s)` position — i.e., the direction of ray `r` must be copied to positions `(r, 0)`, `(r, 1)`, ..., `(r, NS-1)`. That is exactly what the repeat does.

#### The einops string decoded

```
"nrays p d -> nrays nsegments p d"
```

- `nrays` appears in both input and output — this axis passes through unchanged.
- `p d` appears in both — these axes also pass through unchanged.
- `nsegments` appears **only in the output** — this is a new axis being inserted.
- Since `nsegments` is not in the input, einops knows it must create it by **copying the entire input along that new axis** `NS` times.

The `nsegments=NS` keyword argument tells einops the size of that new axis.

---

### `einops.repeat` #2 — Broadcasting Segments Across Rays

<mark></mark>**Location:** `intersect_rays_1d`

```python
segments = einops.repeat(segments, "nsegments p d -> nrays nsegments p d", nrays=NR)
```

#### What the input tensor looks like

`segments` has shape `(NS, 2, 2)` after slicing to xy:

```
segments[0] = [[L10x, L10y], [L20x, L20y]]   ← segment 0
segments[1] = [[L11x, L11y], [L21x, L21y]]   ← segment 1
```

#### What the output tensor looks like

After repeat, shape is `(NR, NS, 2, 2)` — the same final shape as the repeated rays. Now:

```
result[0, 0] = segments[0]   ray 0 paired with segment 0
result[0, 1] = segments[1]   ray 0 paired with segment 1
result[1, 0] = segments[0]   ray 1 paired with segment 0
result[1, 1] = segments[1]   ray 1 paired with segment 1
result[2, 0] = segments[0]   ray 2 paired with segment 0
result[2, 1] = segments[1]   ray 2 paired with segment 1
```

Each individual segment is copied `NR` times along the new `nrays` axis.

#### The einops string decoded

```
"nsegments p d -> nrays nsegments p d"
```

- `nsegments p d` passes through unchanged.
- `nrays` appears only in the output — inserted as a new leading axis.
- The input is copied `NR` times along the `nrays` axis.

#### The combined effect of both repeats

After both operations, you have two tensors of identical shape `(NR, NS, 2, 2)`. At position `[r, s]`:

```
rays[r, s]     = [[Or_x,  Or_y],  [Dr_x,  Dr_y]]   ← ray r's data
segments[r, s] = [[L1s_x, L1s_y], [L2s_x, L2s_y]]  ← segment s's data
```

This is the full Cartesian product. Every ray's data is paired with every segment's data at the corresponding grid position. No Python loop is needed: the subsequent `t.stack`, `t.linalg.det`, `t.linalg.solve` all operate over the leading `(NR, NS)` batch dimensions simultaneously.

```
Before repeat:

  rays      (NR, 2, 2)        segments (NS, 2, 2)

       ray0 ──────────────────────────────────────────────────┐
       ray1 ──────────────────────────────────────────────┐   │
       ray2 ──────────────────────────────────────────┐   │   │
                                                       │   │   │
  After repeat:                                        ▼   ▼   ▼

  rays      (NR=3, NS=2, 2, 2)
            ┌──────────┬──────────┐
    ray 0   │ ray0     │ ray0     │    ← ray 0 copied across both segment slots
            ├──────────┼──────────┤
    ray 1   │ ray1     │ ray1     │    ← ray 1 copied across both segment slots
            ├──────────┼──────────┤
    ray 2   │ ray2     │ ray2     │    ← ray 2 copied across both segment slots
            └──────────┴──────────┘
              seg 0      seg 1

  segments  (NR=3, NS=2, 2, 2)
            ┌──────────┬──────────┐
    ray 0   │ seg0     │ seg1     │    ← segment data varies across columns
            ├──────────┼──────────┤
    ray 1   │ seg0     │ seg1     │    ← same segments, repeated for each ray
            ├──────────┼──────────┤
    ray 2   │ seg0     │ seg1     │
            └──────────┴──────────┘
              seg 0      seg 1

  At every [r, s] position: rays[r,s] + segments[r,s] = the data for pair (ray r, segment s)
```

#### What happens next (without einops)

After the two repeats, the rest of the function is pure PyTorch operating on the `(NR, NS)` batch of 2×2 problems:

```python
O  = rays[:, :, 0]       # (NR, NS, 2) — origin of each ray
D  = rays[:, :, 1]       # (NR, NS, 2) — direction of each ray
L1 = segments[:, :, 0]   # (NR, NS, 2) — first endpoint of each segment
L2 = segments[:, :, 1]   # (NR, NS, 2) — second endpoint

mat = t.stack([D, L1 - L2], dim=-1)   # (NR, NS, 2, 2) — the 2×2 matrix for each pair
dets = t.linalg.det(mat)              # (NR, NS)       — one determinant per pair
is_singular = dets.abs() < 1e-8       # (NR, NS)       — which pairs have no solution
mat[is_singular] = t.eye(2)           # replace singular matrices to avoid solve failure

vec = L1 - O                          # (NR, NS, 2)    — right-hand side
sol = t.linalg.solve(mat, vec)        # (NR, NS, 2)    — [u, v] for each pair
u, v = sol[..., 0], sol[..., 1]      # (NR, NS) each

# A pair (r, s) is a valid intersection when the solution is in range AND the matrix was non-singular
valid = (u >= 0) & (v >= 0) & (v <= 1) & ~is_singular   # (NR, NS)

# For each ray, True if it intersects ANY segment
return valid.any(dim=-1)             # (NR,)
```

The `.any(dim=-1)` at the end reduces the `NS` axis — for each ray, if any of its `NS` intersection tests came back True, the ray hits something. The `dim=-1` is the segment dimension.

---

### `einops.repeat` #3 and #4 — Building the 2D Pixel Grid

**Location:** `make_rays_2d`

```python
def make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit):
    n_pixels = num_pixels_y * num_pixels_z
    ygrid = t.linspace(-y_limit, y_limit, num_pixels_y)   # shape (NY,)
    zgrid = t.linspace(-z_limit, z_limit, num_pixels_z)   # shape (NZ,)

    rays = t.zeros((n_pixels, 2, 3))
    rays[:, 1, 0] = 1                                                         # all rays point in x
    rays[:, 1, 1] = einops.repeat(ygrid, "y -> (y z)", z=num_pixels_z)       # y coordinates
    rays[:, 1, 2] = einops.repeat(zgrid, "z -> (y z)", y=num_pixels_y)       # z coordinates
    return rays
```

#### The conceptual problem

The screen is a 2D grid of `NY × NZ` pixels. Each pixel `(i, j)` corresponds to a ray pointing toward `(x=1, y=ygrid[i], z=zgrid[j])`. But the output `rays` tensor is 1D in its first axis — it has `NY * NZ` rows in order. You need to "unroll" a 2D grid into a 1D list, and the unrolling convention matters: rows vary slowly (y index), columns vary fast (z index).

#### `einops.repeat(ygrid, "y -> (y z)", z=num_pixels_z)`

**Input:** `ygrid` shape `(NY,)` — the NY distinct y-values for the pixel rows.

```
ygrid = [-0.3, -0.1, +0.1, +0.3]    (NY=4 example)
```

**Output:** shape `(NY * NZ,)` — the y-value for each of the `NY*NZ` rays in order.

The string `"y -> (y z)"` with `z=NZ` means: for each element along the `y` axis, emit `NZ` copies before moving to the next `y` element. The `(y z)` merge then flattens those two axes into one.

```
ygrid = [-0.3, -0.1, +0.1, +0.3]   (NY=4, NZ=3)

Expand to (NY, NZ):
  [[-0.3, -0.3, -0.3],
   [-0.1, -0.1, -0.1],
   [+0.1, +0.1, +0.1],
   [+0.3, +0.3, +0.3]]

Flatten (y z) → 1D:
  [-0.3, -0.3, -0.3, -0.1, -0.1, -0.1, +0.1, +0.1, +0.1, +0.3, +0.3, +0.3]
   ←── row 0, 3 pixels ──→  ←── row 1, 3 pixels ──→  ...
```

Each y-value is repeated NZ times consecutively, one block per pixel-row.

#### `einops.repeat(zgrid, "z -> (y z)", y=num_pixels_y)`

**Input:** `zgrid` shape `(NZ,)` — the NZ distinct z-values for the pixel columns.

```
zgrid = [-0.3, 0.0, +0.3]    (NZ=3 example)
```

**Output:** shape `(NY * NZ,)` — the z-value for each ray.

The string `"z -> (y z)"` with `y=NY` means: repeat the entire `z` sequence `NY` times (one repetition per row of the grid), then flatten.

```
zgrid = [-0.3, 0.0, +0.3]   (NY=4, NZ=3)

Expand to (NY, NZ):
  [[-0.3,  0.0, +0.3],
   [-0.3,  0.0, +0.3],
   [-0.3,  0.0, +0.3],
   [-0.3,  0.0, +0.3]]

Flatten (y z) → 1D:
  [-0.3, 0.0, +0.3, -0.3, 0.0, +0.3, -0.3, 0.0, +0.3, -0.3, 0.0, +0.3]
   ←── row 0 ──→    ←── row 1 ──→    ←── row 2 ──→    ←── row 3 ──→
```

The z sequence cycles through all NZ values once per row.

#### The combined effect: a row-major grid unrolled to 1D

At ray index `r = i * NZ + j` (the `r`-th ray in row-major order):

```
rays[r, 1, 1] = ygrid[i]   ← y-coordinate from repeat(ygrid, "y -> (y z)")
rays[r, 1, 2] = zgrid[j]   ← z-coordinate from repeat(zgrid, "z -> (y z)")
```

The two repeats together assign coordinates to every pixel in the grid. This is directly analogous to exercise (6) from section 0.0, where the batch axis was split into `(b1 b2)` to describe a 2D grid of images. Here the pixel grid is described by `(y z)` and then flattened into the 1D ray list.

```
Pixel grid (NY=4, NZ=3):

          z=−0.3    z=0.0    z=+0.3
         ┌────────┬────────┬────────┐
y=−0.3   │ ray 0  │ ray 1  │ ray 2  │
         ├────────┼────────┼────────┤
y=−0.1   │ ray 3  │ ray 4  │ ray 5  │
         ├────────┼────────┼────────┤
y=+0.1   │ ray 6  │ ray 7  │ ray 8  │
         ├────────┼────────┼────────┤
y=+0.3   │ ray 9  │ ray 10 │ ray 11 │
         └────────┴────────┴────────┘

rays[:, 1, 1] = [-0.3,-0.3,-0.3, -0.1,-0.1,-0.1, +0.1,+0.1,+0.1, +0.3,+0.3,+0.3]
                 ← y constant within each row, changes between rows

rays[:, 1, 2] = [-0.3,0.0,+0.3, -0.3,0.0,+0.3, -0.3,0.0,+0.3, -0.3,0.0,+0.3]
                 ← z cycles through all values for each row
```

---

---

## Section 2 Code Walkthrough — All Segments

Section 2 introduces five distinct code blocks. The einops operations were already covered in depth above; this walkthrough focuses on every surrounding line — the slicing, the singular-matrix trick, the logical reduction, and the test calls — so the complete function is fully accounted for.

---

#### Code Segment 1: `intersect_rays_1d` — full function walkthrough

```python
def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"],
    segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, " nrays"]:
    NR = rays.size(0)
    NS = segments.size(0)

    # Get just the x and y coordinates
    rays = rays[..., :2]
    segments = segments[..., :2]

    # Repeat rays and segments so that we can compute the intersection
    # of every (ray, segment) pair
    rays = einops.repeat(rays, "nrays p d -> nrays nsegments p d", nsegments=NS)
    segments = einops.repeat(segments, "nsegments p d -> nrays nsegments p d", nrays=NR)

    O = rays[:, :, 0]
    D = rays[:, :, 1]

    L_1 = segments[:, :, 0]
    L_2 = segments[:, :, 1]

    mat = t.stack([D, L_1 - L_2], dim=-1)
    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(2)

    vec = L_1 - O
    sol = t.linalg.solve(mat, vec)
    u = sol[..., 0]
    v = sol[..., 1]

    return ((u >= 0) & (v >= 0) & (v <= 1) & ~is_singular).any(dim=-1)
```

**`NR = rays.size(0)` and `NS = segments.size(0)`**

`tensor.size(n)` returns the size of axis `n` as a Python integer. This is equivalent to `tensor.shape[n]` but slightly more explicit. Storing `NR` and `NS` upfront avoids re-reading `.size(0)` repeatedly and makes the intent clear: these are the two dimensions of the problem grid.

**`rays = rays[..., :2]` and `segments = segments[..., :2]`**

The `...` (ellipsis) stands for "all preceding axes, however many there are." Since `rays` is `(NR, 2, 3)`, `rays[..., :2]` is equivalent to `rays[:, :, :2]` — keep all elements of axes 0 and 1, but slice axis 2 to columns 0 and 1 (x and y), dropping column 2 (z). This is the 2D projection: everything in section 2 works in the xy-plane. The ellipsis is preferred over explicit `:, :` because it remains correct if the number of leading axes changes, making the code more robust.

**`O = rays[:, :, 0]` and `D = rays[:, :, 1]`**

After the two einops repeats, `rays` has shape `(NR, NS, 2, 2)`. Indexing `[:, :, 0]` selects point index 0 across all `(NR, NS)` pairs, giving the origin `O` of shape `(NR, NS, 2)`. Similarly `[:, :, 1]` gives direction `D` of shape `(NR, NS, 2)`. This is the batched equivalent of `O, D = ray` from section 1 — instead of unpacking a single `(2,)` origin and direction, you are simultaneously extracting all `NR × NS` origins and directions.

**`mat = t.stack([D, L_1 - L_2], dim=-1)`**

Identical in structure to the section 1 single-pair version, but now operating on `(NR, NS, 2)` tensors instead of `(2,)` vectors. The `dim=-1` stack places `D` and `L_1-L_2` as columns of the result. The output `mat` has shape `(NR, NS, 2, 2)` — a batch of `NR × NS` independent 2×2 matrices, one for each `(ray, segment)` pair.

**`dets = t.linalg.det(mat)`**

`torch.linalg.det` accepts a batch of square matrices and returns one determinant per matrix. Input shape `(NR, NS, 2, 2)` → output shape `(NR, NS)`. Each entry `dets[r, s]` is the determinant of the 2×2 system for pair `(r, s)`. A near-zero determinant means the ray and segment are parallel for that specific pair.

**`is_singular = dets.abs() < 1e-8`**

A boolean mask of shape `(NR, NS)`. True wherever the matrix is singular (determinant effectively zero). The threshold `1e-8` accounts for floating-point rounding — a theoretically zero determinant might be represented as a very small non-zero number. Using `.abs()` handles both positive-near-zero and negative-near-zero cases correctly.

**`mat[is_singular] = t.eye(2)`**

This is the singular-matrix replacement trick. `mat[is_singular]` uses boolean indexing: it selects all `(NR, NS)` positions where `is_singular` is True and returns a sub-tensor of shape `(K, 2, 2)` where K is the number of True entries. Assigning `t.eye(2)` to this sub-tensor replaces every singular 2×2 matrix with the identity matrix. The identity is invertible (det = 1) so the subsequent `linalg.solve` will not raise an error for these entries. The results from solving identity-replaced systems are meaningless geometrically, but they are discarded by the `~is_singular` mask in the final return.

**`sol = t.linalg.solve(mat, vec)`**

`torch.linalg.solve` accepts batched inputs. With `mat` of shape `(NR, NS, 2, 2)` and `vec` of shape `(NR, NS, 2)`, it returns `sol` of shape `(NR, NS, 2)` — the solution `[u, v]` for every pair simultaneously. All `NR × NS` 2×2 systems are solved in one GPU-parallel call.

**`u = sol[..., 0]` and `v = sol[..., 1]`**

Extracts the two unknowns from the last axis of `sol`. Shape of each: `(NR, NS)`. Each `u[r, s]` is the ray parameter for pair `(r, s)`, and `v[r, s]` is the segment parameter.

**`return ((u >= 0) & (v >= 0) & (v <= 1) & ~is_singular).any(dim=-1)`**

This single line contains three steps:

1. **Validity mask** — `(u >= 0) & (v >= 0) & (v <= 1) & ~is_singular` produces a boolean tensor of shape `(NR, NS)`. A cell is True only if the system was non-singular AND both parameters are in the valid geometric range. Note `~is_singular`: even if `u` and `v` happen to satisfy the range conditions after solving an identity-replaced system, the pair is excluded because the original matrix was singular.

2. **`.any(dim=-1)`** — collapses the `NS` (segment) axis by asking "is at least one segment entry True?" for each ray. Returns shape `(NR,)`. This is the logical reduction: for each ray, the result is True if it hits *any* of the NS segments.

3. **Return type** — the return annotation `Bool[Tensor, " nrays"]` confirms the output is a 1D boolean tensor of length NR.

**`tests.test_intersect_rays_1d(intersect_rays_1d)`**
**`tests.test_intersect_rays_1d_special_case(intersect_rays_1d)`**

As in section 1, the test calls immediately validate the implementation. `test_intersect_rays_1d` checks the general case across multiple rays and segments. `test_intersect_rays_1d_special_case` specifically checks that the is_singular path works: it constructs rays parallel to segments and confirms the output is False rather than a crash or a spurious True.

---

#### Code Segment 2: `make_rays_2d` — full function walkthrough

```python
def make_rays_2d(
    num_pixels_y: int, num_pixels_z: int,
    y_limit: float, z_limit: float
) -> Float[Tensor, "nrays 2 3"]:
    n_pixels = num_pixels_y * num_pixels_z
    ygrid = t.linspace(-y_limit, y_limit, num_pixels_y)
    zgrid = t.linspace(-z_limit, z_limit, num_pixels_z)
    rays = t.zeros((n_pixels, 2, 3), dtype=t.float32)
    rays[:, 1, 0] = 1
    rays[:, 1, 1] = einops.repeat(ygrid, "y -> (y z)", z=num_pixels_z)
    rays[:, 1, 2] = einops.repeat(zgrid, "z -> (y z)", y=num_pixels_y)
    return rays
```

**`n_pixels = num_pixels_y * num_pixels_z`**

The total number of rays equals the total number of pixels in the 2D grid. A 10×10 screen requires 100 rays, a 120×120 screen requires 14,400. This product is computed once and used as the leading dimension of the output tensor.

**`ygrid = t.linspace(-y_limit, y_limit, num_pixels_y)`**

Creates the y-coordinates for each pixel row — `num_pixels_y` evenly spaced values from `-y_limit` to `+y_limit`. Shape `(NY,)`. These are the y-positions on the screen at `x=1` that the camera is pointing at.

**`zgrid = t.linspace(-z_limit, z_limit, num_pixels_z)`**

Same for z-coordinates. Shape `(NZ,)`. Together `ygrid` and `zgrid` define the 2D grid of target positions.

**`rays = t.zeros((n_pixels, 2, 3), dtype=t.float32)`**

Allocates the `(NY*NZ, 2, 3)` output tensor. The axis contract is identical to `make_rays_1d`: axis 0 is the ray index, axis 1 distinguishes origin (0) from direction (1), axis 2 is the spatial coordinate. All origins remain zero (the camera is at the origin).

**`rays[:, 1, 0] = 1`**

Sets the x-component of every direction to 1. Identical to the `make_rays_1d` line. All rays point in the positive x direction — the screen is at `x=1`.

**`rays[:, 1, 1] = einops.repeat(ygrid, "y -> (y z)", z=num_pixels_z)`**

The einops repeat tile each y-value `NZ` times consecutively, then flattens. The result is a `(NY*NZ,)` 1D sequence where the y-coordinate is constant within each row-block of NZ consecutive entries. This is assigned into the y-component of every direction (`[:, 1, 1]` — all rays, direction point, y-dimension).

**`rays[:, 1, 2] = einops.repeat(zgrid, "z -> (y z)", y=num_pixels_y)`**

Tiles the entire z-sequence `NY` times, then flattens. Result is `(NY*NZ,)` where the z-coordinates cycle through all NZ values for every row. Assigned into the z-component of every direction.

**Return annotation `Float[Tensor, "nrays 2 3"]`**

Documents that the output is a `(NR, 2, 3)` float tensor — the same shape contract as `make_rays_1d`, just with a different number of rays.

**Rendering call:**

```python
rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)
```

As in section 1, the visualisation step immediately follows the function. With a 10×10 grid of 100 rays, the 3D plot should show a pyramid of lines fanning out from the origin in both y and z directions simultaneously. This is the visual sanity check that the `(y z)` einops unrolling produced the correct row-major ordering.

---

#### Code Segment 3: The 5 Tips (conceptual segments, not exercise functions)

The five tips in the notebook are markdown/prose cells with embedded code snippets that demonstrate PyTorch features. They are not exercise functions — you do not implement them — but they introduce syntax that the actual exercises rely on.

**Tip 1 — Elementwise logical operators (`&`, `|`, `~`)**

```python
# Correct: use & | ~ on tensors
(u >= 0) & (v >= 0) & (v <= 1)

# Wrong: Python's 'and' coerces tensors to bool → exception
u >= 0 and v >= 0
```

Python's `and` keyword calls `bool()` on each operand. For a tensor with more than one element, `bool()` raises `RuntimeError: Boolean value of Tensor with more than one element is ambiguous`. The `&` operator calls `torch.__and__` element-wise instead. Similarly `|` for OR and `~` for NOT. Operator precedence matters: `v >= 0 & v <= 1` is parsed as `v >= (0 & v) <= 1` which is nonsensical — always wrap comparison sub-expressions in parentheses: `(v >= 0) & (v <= 1)`.

**Tip 2 — `einops.repeat`**

```python
x = t.randn(4, 3)
x_repeated = einops.repeat(x, 'a b -> a b c', c=2)
assert x_repeated.shape == (4, 3, 2)
```

A preview of the primary tool used in `intersect_rays_1d`. The example shows adding a new trailing axis `c` of size 2, resulting in a copy of `x` stacked along that new dimension.

**Tip 3 — Logical reductions (`.any()`, `.all()`)**

```python
# Reduce along the segment axis to get one bool per ray
valid_any_segment = valid_mask.any(dim=-1)   # shape (NR,)
valid_all_segments = valid_mask.all(dim=-1)  # shape (NR,)
```

`.any(dim=d)` collapses axis `d` by asking "is at least one True?" — used in `intersect_rays_1d` to combine the NS per-segment results into one per-ray answer. `.all(dim=d)` collapses by asking "are all True?" — not used in this section but symmetric.

**Tip 4 — Broadcasting**

```python
B = t.ones(4, 3, 2)
A = t.ones(3, 2)
C = A + B   # A is broadcast: shape (3,2) → (4,3,2)
```

PyTorch aligns shapes from the right. `A` of shape `(3, 2)` matches the last two axes of `B` of shape `(4, 3, 2)`, so `A` is implicitly repeated 4 times along a new leading axis. This is what makes `vec = L_1 - O` work in `intersect_rays_1d` — both are shape `(NR, NS, 2)` after the einops repeats, so subtraction is element-wise with no broadcasting needed.

**Tip 5 — Indexing (ellipsis and boolean indexing)**

```python
x[..., 0]          # last axis index 0, all preceding axes intact
mat[is_singular]   # select entries where is_singular is True
mat[is_singular] = t.eye(2)   # assign into those positions
```

The ellipsis `...` generalises `[:, :, :]` to work regardless of how many axes precede the final index. Boolean indexing with `mat[is_singular]` returns a sub-tensor containing only the rows/slabs where the mask is True — and assigning into it modifies the original tensor in place, which is how the identity-replacement trick works.

## Section 3: Triangle Intersection — Scaling to 3D and Full Meshes

### The Mathematical Extension

Moving from line segments to triangles adds one dimension to the linear system. The intersection condition `O + sD = A + u(B-A) + v(C-A)` rearranges to a 3×3 system:

```
[-D | (B-A) | (C-A)] * [s, u, v]^T = O - A
```

The 3×3 matrix has three column vectors: the negated ray direction, and the two edge vectors of the triangle from vertex A. The solution gives:

- `s`: how far along the ray the intersection is (must be `>= 0` to be in front of the camera)
- `u`, `v`: barycentric coordinates within the triangle plane (must satisfy `u>=0, v>=0, u+v<=1` to be inside the triangle)

The same singular-matrix trick applies: if `det(mat) ≈ 0`, the ray is parallel to the triangle's plane and there is no intersection.

---

### `einops.repeat` #5 — Expanding Triangle Vertices Across Rays

**Location:** `raytrace_triangle`

```python
A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)
```

#### What the input tensor looks like

`triangle` has shape `(3, 3)`:

```
triangle = [[Ax, Ay, Az],   ← vertex A
            [Bx, By, Bz],   ← vertex B
            [Cx, Cy, Cz]]   ← vertex C

axis 0: pts   — which vertex (0=A, 1=B, 2=C)
axis 1: dims  — spatial coordinate (x, y, z)
```

This is a single fixed triangle. All `NR` rays are being tested against *this one* triangle.

#### What the output tensor looks like

After repeat, shape is `(3, NR, 3)`:

```
axis 0: pts   — which vertex (0=A, 1=B, 2=C)  — unchanged
axis 1: NR    — which ray (NEW axis)
axis 2: dims  — spatial coordinate             — unchanged
```

The string `"pts dims -> pts NR dims"` inserts a new `NR` axis between `pts` and `dims`. The entire input is copied `NR` times along that axis, so `result[p, r, :] == triangle[p, :]` for every ray `r`.

#### The Python unpacking trick

After the repeat, unpacking with `A, B, C = ...` splits along `axis 0` (the pts axis):

```
A = result[0]   shape (NR, 3)   vertex A repeated NR times
B = result[1]   shape (NR, 3)   vertex B repeated NR times
C = result[2]   shape (NR, 3)   vertex C repeated NR times
```

Now `A[r]` is vertex A expressed in the frame of ray `r` — it's the same vertex for all `r`, but having it repeated means the arithmetic `B - A`, `C - A`, `O - A` can be computed with straightforward broadcasting across the `(NR, 3)` shape.

#### Why not just use broadcasting directly?

You might wonder: couldn't you subtract `triangle[0]` (shape `(3,)`) from `O` (shape `(NR, 3)`) using broadcasting? Yes — PyTorch broadcasts `(3,)` against `(NR, 3)` by aligning from the right. But the `t.stack([-D, B-A, C-A], dim=-1)` call requires all three column vectors to have the same shape `(NR, 3)`, and you need them in a form that makes their geometric role explicit. The repeat makes `B-A` and `C-A` unambiguously `(NR, 3)` tensors representing the two edge vectors evaluated for every ray simultaneously. The resulting `mat` is `(NR, 3, 3)` — a batch of NR independent 3×3 matrices, one per ray, which `t.linalg.solve` handles natively.

---

### `einops.repeat` #6 and #7 — The Full Cartesian Product: Rays × Triangles

**Location:** `raytrace_mesh`

This is the most complex einops usage in the section. The function must test `NR` rays against `NT` triangles simultaneously, returning the minimum intersection distance for each ray. This requires creating a `(NR, NT)` grid of problems — analogous to the `(NR, NS)` grid in section 2, but now with 3D triangles instead of 2D segments, and with a different post-processing step (min distance instead of boolean any).

```python
triangles = einops.repeat(triangles, "NT pts dims -> pts NR NT dims", NR=NR)
rays      = einops.repeat(rays,      "NR pts dims -> pts NR NT dims", NT=NT)
A, B, C   = triangles    # each shape (NR, NT, 3)
O, D      = rays         # each shape (NR, NT, 3)
```

#### Input shapes

```
triangles: (NT, 3, 3)    — NT triangles, each with 3 vertices, each in 3D
rays:      (NR, 2, 3)    — NR rays, each with origin + direction, each in 3D
```

#### `einops.repeat(triangles, "NT pts dims -> pts NR NT dims", NR=NR)`

The output shape is `(3, NR, NT, 3)`, rearranged so that:

- `pts` (the vertex axis, size 3) is first, enabling the `A, B, C = ...` unpacking.
- `NR` is a new axis inserted between `pts` and `NT`.
- `NT` and `dims` pass through.

The data transformation: each of the `NT` triangles is copied `NR` times along the new `NR` axis. So `result[p, r, t_idx, :]` is vertex `p` of triangle `t_idx`, repeated for ray `r`.

```
Before: triangles (NT=3 example)
  tri0: [[A0x,A0y,A0z], [B0x,B0y,B0z], [C0x,C0y,C0z]]
  tri1: [[A1x,A1y,A1z], [B1x,B1y,B1z], [C1x,C1y,C1z]]
  tri2: [[A2x,A2y,A2z], [B2x,B2y,B2z], [C2x,C2y,C2z]]

After repeat with NR=2, then A = result[0]:
  shape (NR=2, NT=3, 3)

         tri0            tri1            tri2
  ray0 [ [A0x,A0y,A0z], [A1x,A1y,A1z], [A2x,A2y,A2z] ]
  ray1 [ [A0x,A0y,A0z], [A1x,A1y,A1z], [A2x,A2y,A2z] ]
       ↑ identical rows — each ray gets the same triangle vertices
```

#### `einops.repeat(rays, "NR pts dims -> pts NR NT dims", NT=NT)`

The output shape is `(2, NR, NT, 3)`, rearranged so that:

- `pts` (size 2, origin/direction) is first, enabling `O, D = ...` unpacking.
- `NR` passes through.
- `NT` is a new axis inserted between `NR` and `dims`.

The data transformation: each of the `NR` rays is copied `NT` times along the new `NT` axis. So `result[p, r, t_idx, :]` is the origin or direction of ray `r`, repeated for triangle `t_idx`.

```
Before: rays (NR=2 example)
  ray0: [[O0x,O0y,O0z], [D0x,D0y,D0z]]
  ray1: [[O1x,O1y,O1z], [D1x,D1y,D1z]]

After repeat with NT=3, then O = result[0]:
  shape (NR=2, NT=3, 3)

         tri0            tri1            tri2
  ray0 [ [O0x,O0y,O0z], [O0x,O0y,O0z], [O0x,O0y,O0z] ]
  ray1 [ [O1x,O1y,O1z], [O1x,O1y,O1z], [O1x,O1y,O1z] ]
       ↑ identical columns — each triangle gets tested against the same ray origin
```

#### The combined grid

After both repeats and unpacking, you have six tensors all of shape `(NR, NT, 3)`:

```
A[r, t]   vertex A of triangle t, for testing against ray r
B[r, t]   vertex B of triangle t, for testing against ray r
C[r, t]   vertex C of triangle t, for testing against ray r
O[r, t]   origin of ray r, for testing against triangle t
D[r, t]   direction of ray r, for testing against triangle t
```

At every `(r, t)` position, you have exactly the data needed to solve the 3×3 linear system for that specific `(ray, triangle)` pair. No loops, no indexing tricks — just arithmetic on identically-shaped tensors.

```
The (NR, NT) grid of intersection problems:

                 tri0       tri1       tri2   ...  tri(NT-1)
              ┌─────────┬─────────┬─────────┬─────┬─────────┐
  ray 0       │ solve   │ solve   │ solve   │ ... │ solve   │
              ├─────────┼─────────┼─────────┼─────┼─────────┤
  ray 1       │ solve   │ solve   │ solve   │ ... │ solve   │
              ├─────────┼─────────┼─────────┼─────┼─────────┤
  ...         │ ...     │ ...     │ ...     │ ... │ ...     │
              ├─────────┼─────────┼─────────┼─────┼─────────┤
  ray (NR-1)  │ solve   │ solve   │ solve   │ ... │ solve   │
              └─────────┴─────────┴─────────┴─────┴─────────┘

  All NR × NT cells solved simultaneously by t.linalg.solve on a (NR, NT, 3, 3) batch.
```

---

### `einops.reduce` — Collapsing Triangles to Minimum Distance

**Location:** `raytrace_mesh`

```python
s[~intersects] = float("inf")
return einops.reduce(s, "NR NT -> NR", "min")
```

#### What `s` contains at this point

`s` is the distance variable from the linear solve, shape `(NR, NT)`. Each entry `s[r, t]` is the distance along ray `r` to the plane of triangle `t`. But many entries are invalid:

- If `is_singular[r, t]` is True, the matrix was degenerate (ray parallel to triangle plane) — no intersection.
- If `u[r,t] < 0` or `v[r,t] < 0` or `u[r,t]+v[r,t] > 1`, the ray hits the plane but outside the triangle's boundaries.

The code sets `s[~intersects] = float("inf")` for all invalid pairs, so every `s[r, t]` is either a valid finite distance or `+∞`.

#### What the reduce does

```python
einops.reduce(s, "NR NT -> NR", "min")
```

- **Input:** `s` shape `(NR, NT)` — one distance per `(ray, triangle)` pair.
- **Output:** shape `(NR,)` — one distance per ray.
- **Operation:** `"min"` — take the minimum over the `NT` axis (the triangle axis, which disappears).

For each ray `r`, the result is the distance to the *nearest* triangle that the ray actually intersects. If the ray intersects no triangle at all, every entry in `s[r, :]` is `+∞`, so the minimum is also `+∞`, and the calling code uses `t.isfinite(dists)` to determine which rays hit anything.

#### Connection to section 0.0

This `reduce` is the exact analogue of the max-pooling `reduce` from exercise (8) in the einops prerequisites:

```python
# Exercise (8) — max pooling over a spatial window
einops.reduce(arr, "(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)", "max", h2=2, w2=2, b1=2)

# raytrace_mesh — minimum distance over all triangles
einops.reduce(s, "NR NT -> NR", "min")
```

In exercise (8), the `h2` and `w2` axes are the pooling windows — they are reduced with `"max"` and disappear from the output. Here, `NT` is the "pool" — the set of triangles to compare — and it is reduced with `"min"`, keeping only the closest hit. The conceptual pattern is identical: **collapse a "comparison" axis by applying a reduction function, keeping only the surviving result**.

```
s (NR=4, NT=3):

           tri0    tri1    tri2
  ray 0  [  2.5,   inf,   3.1 ]   → min = 2.5   (hits tri0 first)
  ray 1  [  inf,   inf,   inf ]   → min = inf    (hits nothing)
  ray 2  [  inf,   1.8,   0.9 ]   → min = 0.9   (hits tri2 first)
  ray 3  [  4.0,   4.0,   inf ]   → min = 4.0   (ties on tri0 and tri1)

reduce "NR NT -> NR" "min":
  result = [2.5, inf, 0.9, 4.0]   shape (NR=4,)
```

---

## Section 3 Code Walkthrough — All Segments

Section 3 has four exercise functions and several supporting code blocks. The einops operations within each were covered in detail above; this walkthrough covers every other line.

---

#### Code Segment 1: `triangle_ray_intersects` — single ray, single triangle

```python
Point = Float[Tensor, "points=3"]

def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    s, u, v = t.linalg.solve(t.stack([-D, B - A, C - A], dim=1), O - A)
    return ((s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)).item()
```

**`Point = Float[Tensor, "points=3"]`**

A type alias. `"points=3"` means the axis named `points` has fixed size 3 — this is more specific than just `"points"` (unknown size). It documents that `A`, `B`, `C`, `O`, `D` are all 3D coordinate vectors of exactly length 3.

**`t.stack([-D, B - A, C - A], dim=1)`**

Builds the 3×3 coefficient matrix by stacking three column vectors. Each of `-D`, `B-A`, `C-A` is a 1D tensor of shape `(3,)`. Stacking with `dim=1` (equivalent to `dim=-1` here since the inputs are 1D) places them as columns:

```
column 0: -D      = [-Dx, -Dy, -Dz]   coefficient of s in each equation
column 1: B - A   = [Bx-Ax, By-Ay, Bz-Az]   coefficient of u
column 2: C - A   = [Cx-Ax, Cy-Ay, Cz-Az]   coefficient of v

result shape: (3, 3)
```

Note `-D` rather than `D`: rearranging `O + sD = A + u(B-A) + v(C-A)` moves `sD` to the right, giving `-sD` on the left, so the first column is `-D`.

**`O - A`**

The right-hand side of the system `[-D | B-A | C-A] * [s,u,v]^T = O - A`. Shape `(3,)`.

**`s, u, v = t.linalg.solve(...)`**

Python unpacking along the first (and only) axis. `linalg.solve` returns a `(3,)` tensor; unpacking gives three scalar tensors `s`, `u`, `v`. Note: if the matrix is singular (ray parallel to triangle plane), this raises `RuntimeError` — the single-pair version does not handle this gracefully. That is acceptable here because the function is used as a teaching step only; `raytrace_triangle` handles singularity properly.

**`(s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)`**

The four validity conditions for triangle intersection. Compared to segment intersection which had two conditions (`u >= 0`, `0 <= v <= 1`), triangles require four:

- `s >= 0` — the hit is in front of the camera, not behind
- `u >= 0` — non-negative barycentric coordinate
- `v >= 0` — non-negative barycentric coordinate
- `u + v <= 1` — the point is inside the triangle (the third barycentric weight `w = 1-u-v` must be non-negative)

**`.item()`**

Converts the 0-dimensional boolean tensor to a plain Python `bool`. Required because the function return type is `bool`, not `Tensor`.

---

#### Code Segment 2: `raytrace_triangle` — batched over rays, single triangle

```python
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, " nrays"]:
    NR = rays.size(0)

    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)

    O, D = rays.unbind(dim=1)

    mat: Float[Tensor, "NR 3 3"] = t.stack([-D, B - A, C - A], dim=-1)

    dets: Float[Tensor, "NR"] = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol: Float[Tensor, "NR 3"] = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    return (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular
```

**`O, D = rays.unbind(dim=1)`**

`torch.unbind(dim)` splits a tensor along the given axis and returns a tuple of tensors, each with that axis removed. `rays` has shape `(NR, 2, 3)`; unbinding along `dim=1` splits at the size-2 point axis, giving two tensors each of shape `(NR, 3)`: `O` (all origins) and `D` (all directions). This is the batched equivalent of `O, D = ray` from section 1. It is preferred over `rays[:, 0]` and `rays[:, 1]` because it is explicit about the split being exhaustive.

**`mat = t.stack([-D, B - A, C - A], dim=-1)`**

Now operating on `(NR, 3)` tensors instead of `(3,)` vectors. Each of `-D`, `B-A`, `C-A` has shape `(NR, 3)`. Stacking with `dim=-1` creates columns along the last new axis, producing `mat` of shape `(NR, 3, 3)` — a batch of NR independent 3×3 matrices, one per ray.

**`dets = t.linalg.det(mat)`**

Input `(NR, 3, 3)` → output `(NR,)`. One determinant per ray's matrix. For the 3×3 case, a near-zero determinant means the ray is parallel to the triangle's plane (no intersection possible).

**`mat[is_singular] = t.eye(3)`**

Same singular-replacement trick as section 2, now with a 3×3 identity. The boolean mask `is_singular` has shape `(NR,)`. `mat[is_singular]` selects all singular 3×3 matrices and replaces them with `t.eye(3)`.

**`vec = O - A`**

The right-hand side vector. `O` has shape `(NR, 3)` (one origin per ray) and `A` has shape `(NR, 3)` (the same vertex A repeated NR times by einops). Subtraction is element-wise, giving `vec` of shape `(NR, 3)`.

**`sol = t.linalg.solve(mat, vec)`**

Input: `mat` of shape `(NR, 3, 3)` and `vec` of shape `(NR, 3)`. Output: `sol` of shape `(NR, 3)` — the `[s, u, v]` solution vector for every ray.

**`s, u, v = sol.unbind(dim=-1)`**

Splits the solution along the last axis (size 3: the unknowns), giving three tensors each of shape `(NR,)` — one scalar value per ray for each of `s`, `u`, `v`.

**`return (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular`**

The four validity conditions applied element-wise across all NR rays simultaneously. The `~is_singular` term ensures rays whose matrix was replaced with the identity are always marked False regardless of what `s, u, v` happened to evaluate to. Output shape: `(NR,)` boolean tensor.

---

#### Code Segment 3: `raytrace_mesh` — batched over rays and triangles

```python
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, " nrays"]:
    NR = rays.size(0)
    NT = triangles.size(0)

    triangles = einops.repeat(triangles, "NT pts dims -> pts NR NT dims", NR=NR)
    A, B, C = triangles

    rays = einops.repeat(rays, "NR pts dims -> pts NR NT dims", NT=NT)
    O, D = rays

    mat: Float[Tensor, "NR NT 3 3"] = t.stack([-D, B - A, C - A], dim=-1)
    dets: Float[Tensor, "NR NT"] = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec: Float[Tensor, "NR NT 3"] = O - A

    sol: Float[Tensor, "NR NT 3"] = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(-1)

    s *= D[..., 0]

    intersects = (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular
    s[~intersects] = float("inf")

    return einops.reduce(s, "NR NT -> NR", "min")
```

**`A, B, C = triangles` and `O, D = rays`**

After the einops repeats, `triangles` has shape `(3, NR, NT, 3)` and `rays` has shape `(2, NR, NT, 3)`. Unpacking along axis 0 (Python's default for tuple unpacking of a tensor) splits at the leading size-3 or size-2 axis, yielding `A`, `B`, `C` each of shape `(NR, NT, 3)` and `O`, `D` each of shape `(NR, NT, 3)`. This is why the einops strings rearrange `pts` to be the first axis — so Python unpacking extracts the vertex/point components naturally.

**`mat = t.stack([-D, B - A, C - A], dim=-1)`**

Now operating on `(NR, NT, 3)` tensors. The output `mat` has shape `(NR, NT, 3, 3)` — a batch of `NR × NT` independent 3×3 matrices. The `dim=-1` column-stacking is identical in form to sections 1 and 3's triangle variant; only the batch dimensions in front have changed.

**`dets = t.linalg.det(mat)`**

Input `(NR, NT, 3, 3)` → output `(NR, NT)`. One determinant per `(ray, triangle)` pair.

**`mat[is_singular] = t.eye(3)`**

`is_singular` has shape `(NR, NT)`. Boolean indexing selects all singular 3×3 matrices across the full grid and replaces them with identity.

**`s *= D[..., 0]`**

This is the distance correction. The `s` returned by `linalg.solve` is measured in units of the direction vector `D`, not in world-space distance. To convert: the x-component of the hit point is `Ox + s * Dx`. To get the distance along the x-axis specifically (as the section uses x-distance as its depth metric), multiply `s` by `Dx = D[..., 0]`. The `[..., 0]` selects the x-component from `D` of shape `(NR, NT, 3)`, giving `D[..., 0]` of shape `(NR, NT)`.

**`intersects = (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular`**

Note: `s >= 0` is intentionally absent here. The distance correction `s *= D[..., 0]` may flip the sign of `s` depending on the ray direction. The section uses x-depth as the distance metric and handles directionality through the `float("inf")` masking instead.

**`s[~intersects] = float("inf")`**

For every `(ray, triangle)` pair that did not produce a valid intersection, the distance is set to positive infinity. This ensures that when taking the minimum over triangles, non-intersecting pairs can never "win" — they are always dominated by any finite distance from a real intersection.

**`return einops.reduce(s, "NR NT -> NR", "min")`**

The final collapse. `s` is `(NR, NT)` with finite values for valid intersections and `inf` for everything else. The reduce takes the minimum over the NT axis, returning `(NR,)`. Each entry is either the distance to the nearest triangle the ray hit, or `inf` if the ray hit nothing. The caller uses `t.isfinite(dists)` to separate the two cases for rendering.

---

#### Code Segment 4: Rendering and displaying the mesh

```python
triangles = t.load(section_dir / "pikachu.pt", weights_only=True)

num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])
dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
```

**    ****`t.load(section_dir / "pikachu.pt", weights_only=True)`**

Loads a pre-saved PyTorch tensor from disk. `pikachu.pt` contains the `(412, 3, 3)` float tensor of triangle vertices for the Pikachu mesh — 412 triangles, each with 3 vertices, each in 3D. The `weights_only=True` argument is a security flag introduced in newer PyTorch versions: it restricts deserialization to only tensor data, preventing arbitrary Python code from executing if the file were malicious.

**`rays[:, 0] = t.tensor([-2, 0.0, 0.0])`**

Moves the camera origin to `(-2, 0, 0)` — two units behind the origin along the negative x-axis. The Pikachu mesh is centred near `x=0`, so shooting rays from `x=-2` in the positive x-direction will pass through it. Without this shift, the camera at the origin would be inside the mesh and the rendering would not make sense.

**`dists = raytrace_mesh(rays, triangles)`**

The main call. Returns a `(14400,)` float tensor (120×120 = 14,400 rays) with each entry being the x-distance to the nearest triangle, or `inf` if no triangle was hit.

**`t.isfinite(dists).view(num_pixels_y, num_pixels_z)`**

`t.isfinite` returns a boolean tensor: True where the value is a finite number, False where it is `inf` or `nan`. `.view(120, 120)` reshapes the flat `(14400,)` tensor back into the 2D pixel grid. This is the binary "did this ray hit anything?" image.

**`dists.view(num_pixels_y, num_pixels_z)`**

The depth image — darker pixels are closer (smaller `s`), brighter pixels are farther. Pixels with `inf` appear as bright white or are coloured differently by the colormap.

**`t.stack([intersects, dists_square], dim=0)`**

Creates a `(2, 120, 120)` tensor — two "channels" stacked along a new leading axis. The first channel is the boolean hit map, the second is the depth map. Passing this to `px.imshow` with `facet_col=0` displays both images side by side.

**`origin="lower"`**

Plotly's default image origin is the upper-left corner (y increases downward, as in screen coordinates). `origin="lower"` flips to the mathematical convention (y increases upward), which matches how the rays were constructed with `make_rays_2d`.

---

## The Recurring Pattern: A Unified View

Every einops operation in this section is an instance of one of two patterns. Recognising them makes the code immediately readable.

### Pattern 1: Repeat-to-Broadcast (Cartesian Product Setup)

Used in `intersect_rays_1d`, `raytrace_triangle`, `raytrace_mesh`, and `make_rays_2d`.

**Goal:** Take two tensors that each describe one dimension of a 2D grid of problems, and expand both to the same combined shape so every pair is represented explicitly.

**Template:**

```
tensor_A  (dim_A, ...)   →   repeat "dim_A ... -> dim_A dim_B ..."   →   (dim_A, dim_B, ...)
tensor_B  (dim_B, ...)   →   repeat "dim_B ... -> dim_A dim_B ..."   →   (dim_A, dim_B, ...)
```

After both repeats:

- At position `[a, b]`, `tensor_A[a, b]` is the data for item `a`, and `tensor_B[a, b]` is the data for item `b`.
- All arithmetic between `tensor_A` and `tensor_B` automatically computes results for all `(a, b)` pairs.

| Function            | dim_A          | dim_B               | What's being paired            |
| ------------------- | -------------- | ------------------- | ------------------------------ |
| `intersect_rays_1d` | NR (rays)      | NS (segments)       | every ray with every segment   |
| `raytrace_triangle` | NR (rays)      | — (single triangle) | every ray against one triangle |
| `raytrace_mesh`     | NR (rays)      | NT (triangles)      | every ray with every triangle  |
| `make_rays_2d`      | y (pixel rows) | z (pixel columns)   | every (y,z) pixel position     |

### Pattern 2: Reduce-to-Answer (Collapsing the Comparison Axis)

Used in `raytrace_mesh` (and analogous to max-pooling in exercise (8)).

**Goal:** After computing a result for every pair in the grid, collapse one axis by selecting the "best" value according to a criterion.

**Template:**

```
results  (dim_A, dim_B)   →   reduce "dim_A dim_B -> dim_A" "criterion"   →   (dim_A,)
```

| Function            | Collapsed axis       | Criterion      | Meaning                            |
| ------------------- | -------------------- | -------------- | ---------------------------------- |
| `intersect_rays_1d` | NS (segments)        | `.any(dim=-1)` | does the ray hit *any* segment?    |
| `raytrace_mesh`     | NT (triangles)       | `"min"`        | distance to the *nearest* triangle |
| Exercise (8)        | h2, w2 (pool window) | `"max"`        | brightest pixel in each 2×2 block  |

---

## Summary: What Einops Buys You Here

Without einops, the batching would require either:

1. **Python loops** — functionally correct but orders of magnitude slower, and untranslatable to GPU.
2. **Manual `.unsqueeze().expand()`** calls — achieves the same shapes but is verbose, error-prone (it is easy to expand the wrong axis), and produces code where the geometric intent is invisible.

Einops makes the intent explicit through named axes. When you read:

```python
rays = einops.repeat(rays, "nrays p d -> nrays nsegments p d", nsegments=NS)
```

you immediately know: *rays has a nrays axis, and I am inserting a nsegments axis next to it*. The geometric interpretation — "pair every ray with every segment" — is encoded in the axis names. When you read:

```python
einops.reduce(s, "NR NT -> NR", "min")
```

you immediately know: *collapse the NT axis by taking the minimum, keeping NR*. The geometric interpretation — "for each ray, find the nearest triangle" — follows directly from the axis names.

This is the deeper purpose of einops beyond the image exercises in section 0.0: it provides a notation where **the shape of the data and the shape of the computation are the same thing**, and where every transformation is expressed in terms of what the axes mean, not how the bytes are laid out in memory 
