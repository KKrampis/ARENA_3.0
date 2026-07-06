# Ray Tracing: Context, History, and the Linear Algebra Foundation

---

## Why Einops and Ray Tracing Have No Shared History

Honestly, there is no real shared history between ray tracing and einops — they come from completely different worlds and different eras. Ray tracing as a rendering technique dates back to the 1960s and was formalized by Turner Whitted in 1980 as a way to simulate how light physically bounces through a scene by tracing the path of individual rays from the camera backward to light sources. For decades it was considered too computationally expensive for real-time use, and it was primarily the domain of film studios and scientific visualization, running on dedicated hardware or overnight on CPU farms. It became mainstream in real-time graphics only around 2018 when NVIDIA introduced RT Cores in their Turing GPU architecture. The mathematics — parametric ray equations, barycentric coordinates, linear system solves — has not changed in fifty years.

Einops, by contrast, is a very recent Python library created by Alex Rogozhnikov and released around 2018, designed entirely for deep learning research. Its purpose is to make tensor reshaping and axis manipulation readable through named-axis notation. It was not designed with ray tracing in mind at all. The reason they appear together in this ARENA curriculum is purely pedagogical: ray tracing turns out to be an ideal teaching problem for batched tensor operations because the geometry makes the reason for every reshape concrete and visual. When you write `repeat(rays, "NR pts dims -> NR NT dims")`, the axis name `NT` has an immediate geometric meaning — "one copy per triangle" — that would be invisible in a purely abstract deep learning context. So their pairing here is a curriculum design choice, not a historical connection.

---

## Why Real-Time Ray Tracing Wasn't Available on GPUs Before 2018

Ray tracing wasn't available on GPUs before around 2018 primarily because of how GPUs were architecturally designed. Traditional GPUs were built as massively parallel rasterization machines — they are optimized for a completely different rendering approach where you project 3D geometry onto a 2D screen in a single forward pass, filling in triangles pixel by pixel. Every processing unit on a GPU is designed to do the same simple operation (shade a pixel) across thousands of pixels simultaneously, which maps perfectly onto the regular, predictable math of rasterization. Ray tracing breaks this model entirely because rays scatter in unpredictable directions when they bounce off surfaces — one ray might hit a mirror and reflect, another might hit glass and refract, another might hit a matte wall and stop. This irregular, branching computation pattern is exactly what GPU hardware historically handled poorly. The cores would stall waiting for each other, cache coherence would collapse because rays flying in random directions access memory in random orders, and the theoretical parallelism of the GPU would go largely unused.

The deeper issue was that ray-triangle intersection — the core operation — requires a BVH (bounding volume hierarchy), a tree data structure that lets you quickly narrow down which triangles a ray might hit without testing all of them. Traversing a tree is inherently serial and branch-heavy, the opposite of what GPU shaders are good at. For years researchers found clever software workarounds, but performance was always a fraction of what rasterization achieved. NVIDIA's 2018 Turing architecture introduced dedicated RT Cores — fixed-function hardware units separate from the shader cores — whose sole job is BVH traversal and ray-triangle intersection testing in hardware at fixed cost, freeing the shader cores to handle the shading logic in parallel. This hardware-software co-design, combined with AI-powered denoising (DLSS) to reconstruct a clean image from fewer rays, finally made real-time ray tracing practical. Before that, the math was always known; the hardware simply wasn't shaped for it.

---

## Why Triangles?

Triangles are the universal primitive of 3D graphics for one elegantly simple reason: three points always define exactly one plane. Take any three non-collinear points in 3D space and there is one and only one flat surface passing through all three. This is not true for four or more points — a quad (four-sided polygon) can be slightly warped so its corners don't all sit on the same plane, which creates ambiguity about how to interpolate across its surface and how to test intersections. Triangles have no such ambiguity. This mathematical guarantee means every triangle is always perfectly flat, which makes the ray intersection math clean and exact — the linear system `[-D | B-A | C-A] * [s, u, v]^T = O - A` always has a definite geometric interpretation with no edge cases about which plane to use.

The second reason is that triangles are the simplest polygon that can enclose area. A line (2 points) has no interior. A triangle (3 points) is the minimum shape that does, and any more complex polygon — a pentagon, an irregular blob, a curved surface — can be approximated by breaking it into triangles, a process called tessellation. This means GPU hardware only needs to be optimized for one shape. The entire graphics pipeline, from vertex shaders to rasterizers to RT Cores, is built around the triangle as the atomic unit. Any 3D model, no matter how organic or complex, gets decomposed into triangles before it touches the GPU. The Pikachu in this section is 412 triangles; a modern AAA game character might be 100,000. The shape is always the same — only the count changes — and that uniformity is what allows the hardware to be so efficient.

---

## How Ray Tracing Is Solved via a Linear System

### In This Document

The key insight is that both a ray and any object it might hit — whether a line segment or a triangle — can be written as parametric equations, and "intersection" simply means finding parameter values where both equations describe the same point in space. For a ray and a segment in 2D, you have two equations (`P = O + u*D` and `P = L1 + v*(L2-L1)`) and two unknowns (`u` and `v`). Setting them equal and rearranging the algebra moves all the unknowns to the left and all the known geometry to the right, producing exactly the form `Ax = b` — a matrix of coefficients on the left, a vector of constants on the right, and the unknowns `[u, v]` in the middle.

The matrix `A` has the ray direction `D` as its first column and the segment vector `L1-L2` as its second column; the right-hand side is `L1-O`:

```
[ Dx   L1x-L2x ] [ u ]   [ L1x-Ox ]
[ Dy   L1y-L2y ] [ v ] = [ L1y-Oy ]
```

Calling `torch.linalg.solve(A, b)` recovers `u` and `v`, and you just check whether they fall in the valid ranges (`u >= 0` for the ray, `0 <= v <= 1` for the segment). For triangles the same logic extends to 3D: the ray equation and the barycentric parametrization of the triangle surface produce three scalar equations and three unknowns `(s, u, v)`, giving a 3×3 system with the exact same structure — column-stack the relevant vectors, solve, check validity conditions:

```
[-Dx   (B-A)x   (C-A)x ] [ s ]   [ (O-A)x ]
[-Dy   (B-A)y   (C-A)y ] [ u ] = [ (O-A)y ]
[-Dz   (B-A)z   (C-A)z ] [ v ]   [ (O-A)z ]
```

Here `s` is the distance along the ray to the triangle's plane, and `(u, v)` are barycentric coordinates that determine whether the hit point is inside the triangle (`u >= 0`, `v >= 0`, `u + v <= 1`).

### More Broadly

This is an instance of a much older idea in geometry: intersection problems between lines, planes, and surfaces almost always reduce to linear algebra when the objects themselves are linear (flat). A ray is a linear object — it is a straight half-line. A line segment is linear. A triangle is a flat polygon sitting in a plane. Whenever you intersect two linear objects, the "where do they meet?" question translates directly into "where do these linear equations have a common solution?" — which is the definition of solving a linear system. This is why `torch.linalg.solve` appears so naturally here: it is not a trick specific to ray tracing, it is the general tool for any problem of the form "find the point satisfying multiple linear constraints simultaneously."

The approach breaks down only when objects are curved. A ray hitting a sphere requires solving a quadratic equation because the sphere's surface is nonlinear, giving two possible intersection points (entry and exit) rather than one. For curved surfaces in production renderers, the surface is typically subdivided into tiny flat triangles fine enough that the curvature is imperceptible, bringing the problem back to the triangle-linear system framework used here. So the linear algebra approach in the ARENA exercises is not a simplification for pedagogical convenience — it is the actual foundation of how real-time and offline renderers handle geometry, scaled up to millions of triangles solved in parallel on dedicated hardware.

### The Connection to Batched Tensor Operations

What makes this section particularly powerful as a learning exercise is that the linear system structure maps perfectly onto PyTorch's batched linear algebra. A single `torch.linalg.solve` call accepts not just one `(3, 3)` matrix and one `(3,)` vector, but an entire batch — say `(NR, NT, 3, 3)` matrices and `(NR, NT, 3)` vectors — and solves all systems simultaneously. The einops operations exist precisely to reshape the ray and triangle data into that batch-compatible form. The mathematical template (parametric equations → linear system → solve → check validity) is identical for every `(ray, triangle)` pair; the tensor machinery ensures it runs for all pairs at once rather than one at a time.

```
Single pair (Section 1):         Batched (Section 3):

  A  shape (3, 3)                  A  shape (NR, NT, 3, 3)
  b  shape (3,)                    b  shape (NR, NT, 3)
  x  shape (3,)       →            x  shape (NR, NT, 3)
  1 system solved                  NR × NT systems solved
  by 1 linalg.solve call           by 1 linalg.solve call
```

The jump from one system to millions is not a change in the mathematics — it is purely a change in the shape of the tensors handed to the same function.
