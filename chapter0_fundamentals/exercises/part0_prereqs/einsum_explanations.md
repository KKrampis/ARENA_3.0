# Einstein Summation and Linear Algebra as Tensor Operations

## The Einsum Notation

Einstein summation convention is a compact notation for expressing a wide class of linear algebra operations through a single rule: repeated index names across input tensors imply element-wise multiplication along that axis, and any index that does not appear in the output is summed over. In the `einops.einsum` formulation used throughout this course, the arrays are passed as the first arguments and the index string comes last, with named axes separated by spaces and the `->` separator dividing inputs from the output. The three governing rules are simple: shared index names across inputs mean multiply; absent index names in the output mean sum; and the indices on the right of `->` determine the shape and axis ordering of the result.

These three rules, applied in combination, are sufficient to express virtually every operation in linear algebra and most of the tensor contractions that arise in deep learning. The five exercises that follow — trace, matrix-vector product, matrix-matrix product, inner product, and outer product — cover the canonical cases and together form a complete vocabulary for working with einsum.

---

## The Trace (einsum_trace)

```python
einops.einsum(mat, "i i ->")
```

The trace of a square matrix is the sum of its diagonal elements: `tr(A) = sum_i A[i, i]`. In einsum notation this is written `einops.einsum(mat, "i i ->")`. The two occurrences of `i` on the input side mean that we only ever access positions where the row index equals the column index — the diagonal — and multiply the corresponding elements together (trivially, since there is only one input, this multiplication is just the element itself). The empty right-hand side of `->` means no indices survive into the output, so all selected values are summed to a scalar. No explicit loop, no `torch.diagonal`, no `.sum()` call is needed; the full computation is encoded in the two repeated letters and the empty output specification.

It is worth pausing on what the repeated index `i i` actually forbids: all off-diagonal elements. A single index `i` on its own would access every element of a 1D tensor; two separate indices `i j` would access every element of a 2D matrix. But two *identical* indices on a 2D tensor constrain the access pattern to only those positions where both indices happen to take the same value, which is precisely the main diagonal. This constraint mechanism — using repeated names to enforce equality of indices — is the source of einsum's expressive power.

---

## The Matrix-Vector Product (einsum_mv)

```python
einops.einsum(mat, vec, "i j, j -> i")
```

Multiplying a matrix `A` of shape `(m, n)` by a vector `b` of shape `(n,)` produces a vector `c` of shape `(m,)` where `c[i] = sum_j A[i, j] * b[j]`. In einsum this is `einops.einsum(mat, vec, "i j, j -> i")`. The index `j` appears in both inputs — once as the column of the matrix and once as the position in the vector — so it triggers element-wise multiplication along that axis. The index `j` does not appear in the output, so those products are summed over. The index `i` appears only in the matrix input and in the output, so it passes through unchanged, labelling the rows of the result.

This operation is the atomic building block of neural network computation: every linear layer computes exactly this product (or its batched generalisation) to transform an input vector into an output vector. The einsum string `"i j, j -> i"` makes the mechanics explicit — the shared `j` axis is the contracted dimension along which the dot products are accumulated — which is why understanding einsum at this level provides deep intuition for what a weight matrix actually does during a forward pass.

---

## The Matrix-Matrix Product (einsum_mm)

```python
einops.einsum(mat1, mat2, "i j, j k -> i k")
```

Matrix multiplication of `A` with shape `(m, n)` and `B` with shape `(n, p)` produces `C` of shape `(m, p)` where `C[i, k] = sum_j A[i, j] * B[j, k]`. The einsum string is `"i j, j k -> i k"`. As before, `j` is the contracted dimension: it is shared between both inputs and absent from the output, so the products along `j` are summed. The surviving indices `i` and `k` label the rows of `A` and the columns of `B` respectively, and they appear in the output with that ordering.

Comparing `einsum_mv` and `einsum_mm` makes the generalisation clear: matrix-vector multiplication is the special case where `B` has only one column, so `k` has size 1 and effectively disappears. The same `j`-contraction logic applies. This perspective — matrix multiplication as a sum over a shared contracted index — is the foundation for understanding multi-head attention, where the query-key product `QK^T` is a batched matrix multiplication with an additional `head` index passing through unchanged: `"b h q d, b h k d -> b h q k"`. Recognising the contracted dimension `d` as the key-query alignment axis makes the attention mechanism a direct consequence of the same rule seen here.

---

## The Inner Product (einsum_inner)

```python
einops.einsum(vec1, vec2, "i, i ->")
```

The inner (dot) product of two vectors `u` and `v` of shape `(n,)` is the scalar `sum_i u[i] * v[i]`. In einsum this is `einops.einsum(vec1, vec2, "i, i ->")`. Both inputs share the single index `i`, which means multiply element-wise along that axis; the empty output means sum everything to a scalar. Comparing this to `einsum_trace` reveals a structural parallel: both end with `->` and produce scalars, and both use repeated indices to enforce an element-wise selection before summing. The difference is that `trace` constrains two axes *within* a single 2D tensor to be equal, while `inner` multiplies two *separate* 1D tensors element-wise before summing.

The inner product is also the building block for cosine similarity (exercise C2): once two vectors are normalised to unit length, their inner product is their cosine similarity. More generally, every row of a matrix-vector product is an inner product between a row of the matrix and the vector, which is why `"i j, j -> i"` and `"i, i ->"` look so similar — the `mv` case just maps over the `i` axis rather than consuming it.

### Worked example

```
u = [1, 2, 3]
v = [4, 5, 6]

Step 1 — multiply element-wise (shared index i):

  i=0:  u[0] * v[0]  =  1 * 4  =   4
  i=1:  u[1] * v[1]  =  2 * 5  =  10
  i=2:  u[2] * v[2]  =  3 * 6  =  18

Step 2 — sum (i absent from output "->"):

  result = 4 + 10 + 18 = 32   shape: scalar
```

Equivalently in closed form: `u · v = (1×4) + (2×5) + (3×6) = 32`.
The result is a single number measuring the "alignment" between the two vectors.

---

## The Outer Product (einsum_outer)

```python
einops.einsum(vec1, vec2, "i, j -> i j")
```

The outer product of vectors `u` of shape `(m,)` and `v` of shape `(n,)` produces a matrix of shape `(m, n)` where `out[i, j] = u[i] * v[j]`. In einsum this is `einops.einsum(vec1, vec2, "i, j -> i j")`. Here the indices `i` and `j` are *distinct* across the two inputs — neither is shared — so there is no contraction and no summation. Every combination of one element from `vec1` and one from `vec2` is multiplied, and the result is laid out as a 2D matrix indexed by both. The output `-> i j` determines that the first axis corresponds to `vec1` and the second to `vec2`.

The outer product is in some sense the opposite of the inner product: the inner product takes two vectors of the same length and collapses them to a scalar by summing all pairwise products along the shared axis, while the outer product takes two vectors of any lengths and expands them to a matrix by keeping all pairwise products without any summation. This duality — contraction versus expansion — is the core tension in all tensor operations, and einsum makes it visible through the presence or absence of shared index names. In neural networks, outer products arise in low-rank weight decompositions, in computing gradients of linear layers, and in the key-value update mechanism of certain memory models.

### Worked example

```
u = [1, 2, 3]     shape (3,)   → labels rows    (index i)
v = [4, 5, 6]     shape (3,)   → labels columns (index j)

For every (i, j) pair: out[i, j] = u[i] * v[j]

         v[0]=4   v[1]=5   v[2]=6
         ───────  ───────  ───────
u[0]=1 │  1×4=4   1×5=5   1×6=6
u[1]=2 │  2×4=8   2×5=10  2×6=12
u[2]=3 │  3×4=12  3×5=15  3×6=18

result = [[ 4,  5,  6],
          [ 8, 10, 12],
          [12, 15, 18]]    shape (3, 3)
```

Notice that every row is a scalar multiple of `v` (row `i` = `u[i] * v`), and every
column is a scalar multiple of `u` (column `j` = `v[j] * u`). This means the outer
product always produces a **rank-1 matrix** — a matrix whose rows (and columns) are
all proportional to each other.

### Inner vs outer — side by side

```
              INNER  "i, i ->"          OUTER  "i, j -> i j"
              ──────────────────        ──────────────────────
u             [1, 2, 3]                [1, 2, 3]
v             [4, 5, 6]                [4, 5, 6]
shared index  i  (same axis, same dim) none  (i and j are independent)
operation     element-wise multiply    all pairwise products
then          sum everything           keep everything
output shape  scalar  ()               matrix  (3, 3)
result        32                       [[ 4,  5,  6],
                                        [ 8, 10, 12],
                                        [12, 15, 18]]
```

---

## The Unified View

Taken together, these five operations reveal that the `->` notation in einsum divides indices into two categories: those that are **contracted** (shared across inputs, absent from output, triggering summation) and those that are **free** (present in the output, passing through or expanding). The trace and inner product contract all indices to scalars; the matrix-vector product contracts one axis and keeps one; the matrix-matrix product contracts one axis and keeps two; the outer product contracts nothing and keeps two.

```
Operation   String              Contracted   Free    Output shape
─────────────────────────────────────────────────────────────────
trace       "i i ->"            i            —       ()
inner       "i, i ->"           i            —       ()
mv          "i j, j -> i"       j            i       (m,)
mm          "i j, j k -> i k"   j            i, k    (m, p)
outer       "i, j -> i j"       —            i, j    (m, n)
```

Every more complex operation in the transformer chapters — batched attention, multi-head projections, weight-tied output embeddings — is a combination of these same two mechanisms applied over additional batch and head dimensions. Fluency with einsum therefore means reading any tensor contraction directly from its index string without needing to mentally simulate loops or intermediate shapes.
