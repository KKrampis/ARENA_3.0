# Broadcasting & Tensor Manipulation Exercises — Detailed Explanations

---

## (A1) rearrange

```python
def rearrange_1() -> Tensor:
    return einops.rearrange(t.arange(3, 9), "(h w) -> h w", h=3, w=2)
# [[3, 4], [5, 6], [7, 8]]
```

`t.arange(3, 9)` produces a flat 1D tensor `[3, 4, 5, 6, 7, 8]` of shape `(6,)`. The einops pattern `"(h w) -> h w"` with `h=3, w=2` tells einops to interpret that flat dimension as the product of two axes and split it — the first 2 elements form row 0, the next 2 form row 1, and the last 2 form row 2. This is essentially a reshape from `(6,)` to `(3, 2)`, but expressed in named-axis notation that makes the intended shape explicit without relying on raw integer tuples.

The key insight is that parenthesised axes on the *input* side mean "split this dimension according to the sizes given". You must provide enough size hints to let einops infer all dimensions unambiguously — here `h=3` is supplied and `w` is inferred as `6 / 3 = 2`. This is the inverse of the merge operation `(h w)` on the *output* side seen in the image exercises: input-side parentheses split, output-side parentheses merge.

---

## (A2) rearrange

```python
def rearrange_2() -> Tensor:
    return einops.rearrange(t.arange(1, 7), "(h w) -> h w", h=2, w=3)
# [[1, 2, 3], [4, 5, 6]]
```

This follows the same pattern as A1: `t.arange(1, 7)` gives `[1, 2, 3, 4, 5, 6]` of shape `(6,)`, which is reshaped into `(2, 3)` by splitting the flat dimension into `h=2` rows and `w=3` columns. Elements fill in row-major (C-contiguous) order, so the first 3 values populate row 0 and the next 3 populate row 1.

Comparing A1 and A2 highlights that the same 6-element sequence can be reshaped into `(3, 2)` or `(2, 3)` depending solely on which factor you assign to `h` versus `w`. This is a foundational idea for the rest of the course: tensor shape is a convention, not an intrinsic property of the data, and einops makes shape transformations explicit and human-readable rather than hiding them behind magic numbers in a `view()` or `reshape()` call.

---

## (B1) temperature average

```python
def temperatures_average(temps: Tensor) -> Tensor:
    return einops.reduce(temps, "(h 7) -> h", "mean")
```

`temps` is a 1D tensor of daily temperatures whose length is a multiple of 7. The pattern `"(h 7) -> h"` first conceptually splits the flat dimension into two axes: `h` weeks of 7 days each. The constant `7` is a literal size hint inside the parentheses, not a named dimension, so einops knows each group is exactly 7 elements long and infers `h` automatically. The `"mean"` reduction then collapses the 7-day axis, leaving one average value per week.

This demonstrates one of einops's most elegant features: you can embed literal sizes inside the split notation to express "group by fixed-size windows" in a single function call. The equivalent PyTorch would require a `view(h, 7).mean(dim=1)` — two operations with implicit intermediate shapes. The einops version keeps everything in one expression and makes the grouping logic self-documenting.

---

## (B2) temperature difference

```python
def temperatures_differences(temps: Tensor) -> Tensor:
    avg = einops.reduce(temps, "(w 7) -> w", "mean")
    return temps - einops.repeat(avg, "w -> (w 7)")
```

After computing the per-week averages (shape `(num_weeks,)`), we need to subtract each week's average from every day within that week. The tensors `temps` (shape `(14,)`) and `avg` (shape `(2,)`) are not directly broadcastable because PyTorch aligns dimensions from the right: it would try to match a size-14 axis against a size-2 axis, which fails. The fix is to use `einops.repeat(avg, "w -> (w 7)")`, which expands the averages back to shape `(14,)` by repeating each weekly value 7 times consecutively.

This is the inverse of the B1 reduction: where B1 collapsed `(w 7) -> w`, here we expand `w -> (w 7)`. The pattern shows the clean symmetry between `reduce` and `repeat` in einops — you can think of them as encoding and decoding the same grouping structure. An alternative would be to reshape `temps` to `(2, 7)`, subtract `avg.unsqueeze(1)` (which broadcasts along the day axis), then flatten back — but the einops approach expresses the intent more directly.

---

## (B3) temperature normalized

```python
def temperatures_normalized(temps: Tensor) -> Tensor:
    avg = einops.reduce(temps, "(w 7) -> w", "mean")
    std = einops.reduce(temps, "(h 7) -> h", t.std)
    return (temps - einops.repeat(avg, "w -> (w 7)")) / einops.repeat(std, "w -> (w 7)")
```

This extends B2 by also dividing by each week's standard deviation, producing a z-score for every day relative to its own week. The standard deviation is computed by passing `t.std` as a callable into `einops.reduce` instead of a string like `"mean"`. Einops supports any function that reduces a tensor along a dimension, which makes it highly flexible for custom aggregations beyond the built-ins.

Note that `t.std` uses Bessel's correction (divides by `n-1`) by default, which is appropriate for sample statistics. The normalization then follows the usual z-score formula: subtract the mean and divide by the standard deviation, both broadcast back to the original shape via `einops.repeat`. The result is that days warmer than average for their week are positive, cooler days are negative, and the scale is in units of standard deviations — a common pre-processing step before feeding time-series data into a model.

---

## (C1) normalize rows

```python
def normalize_rows(matrix: Tensor) -> Tensor:
    row_norms = matrix.norm(dim=1, keepdim=True)
    return matrix / row_norms
```

`matrix.norm(dim=1)` computes the L2 norm (Euclidean length) of each row, producing a 1D tensor of shape `(m,)`. Without `keepdim=True`, dividing `matrix` (shape `(m, n)`) by this 1D tensor would require either an explicit `unsqueeze` or relying on PyTorch's right-alignment broadcasting, where `(m,)` aligns with the *last* axis `n`, not the row axis `m` — producing wrong results silently. Setting `keepdim=True` retains the reduced dimension as a size-1 axis, giving shape `(m, 1)`, which then broadcasts cleanly across all `n` columns of each row.

This is one of the most important practical lessons in tensor programming: when you reduce along a dimension you intend to broadcast back over, always use `keepdim=True`. It prevents the subtle shape-misalignment bugs that are among the most common sources of silent errors in PyTorch code. The L2-normalized matrix has the property that `matrix[i] @ matrix[i] == 1.0` for every row `i`, which is the foundation of cosine similarity.

---

## (C2) pairwise cosine similarity

```python
def cos_sim_matrix(matrix: Tensor) -> Tensor:
    matrix_normalized = normalize_rows(matrix)
    return matrix_normalized @ matrix_normalized.T
```

Cosine similarity between two unit vectors is simply their dot product. After normalizing every row to unit length with `normalize_rows`, the matrix multiplication `matrix_normalized @ matrix_normalized.T` computes all pairwise dot products at once. For an input of shape `(m, n)`, the normalized matrix is also `(m, n)` and its transpose is `(n, m)`, so the result is `(m, m)` where entry `[i, j]` is the cosine similarity between row `i` and row `j`. Diagonal entries are always `1.0` (a vector is perfectly similar to itself), and off-diagonal entries range from -1 to 1.

This is a beautiful example of how linear algebra and broadcasting interact in deep learning. Computing the full similarity matrix in a loop would require `O(m²)` separate dot products; the batched matrix multiply achieves the same result in a single highly-optimised BLAS call. Cosine similarity matrices appear throughout the course — most prominently in attention mechanisms, where the query-key dot product matrix is exactly this structure (before scaling and softmax).

---

## (D) sample distribution

```python
def sample_distribution(probs: Tensor, n: int) -> Tensor:
    return (t.rand(n, 1) > t.cumsum(probs, dim=0)).sum(dim=-1)
```

The cumulative sum `t.cumsum(probs, dim=0)` of a probability distribution creates a sequence of thresholds: if `probs = [0.2, 0.3, 0.5]` then `cumsum = [0.2, 0.5, 1.0]`. A uniform random value `v` drawn from `[0, 1)` falls below threshold `k` with exactly the probability that class `k` should be selected. Counting how many thresholds `v` exceeds (i.e., `sum(v > cumsum)`) therefore maps `v` to a sample index from the original distribution. `t.rand(n, 1)` generates `n` such values, each in its own row, and the shape `(n, 1)` broadcasts against `cumsum` (shape `(k,)`) to produce an `(n, k)` boolean matrix via the comparison `>`.

Summing the boolean matrix along `dim=-1` gives each of the `n` samples as a scalar index in `[0, k)`. The brilliance of this approach is that it avoids explicit Python loops entirely: all `n` samples are drawn simultaneously through broadcasting. This vectorised inverse-CDF (quantile) method is numerically exact and runs in O(n·k) time on the GPU. It also illustrates a general principle: whenever you find yourself writing `for i in range(n): result[i] = f(...)`, look for a broadcasting reformulation.

---

## (E) classifier accuracy

```python
def classifier_accuracy(scores: Tensor, true_classes: Tensor) -> Tensor:
    return (scores.argmax(dim=1) == true_classes).float().mean()
```

`scores.argmax(dim=1)` finds the column index of the highest score in each row of the `(batch, n_classes)` matrix, returning a 1D tensor of shape `(batch,)` containing the predicted class for each example. Comparing this element-wise to `true_classes` (also shape `(batch,)`) produces a boolean tensor that is `True` wherever the model's top prediction matches the ground truth. Calling `.float()` converts `True → 1.0` and `False → 0.0`, and `.mean()` then computes the fraction of correct predictions — the standard definition of top-1 accuracy.

This one-liner demonstrates that evaluation metrics in PyTorch are just tensor operations like any other. There is no need for special metric libraries for simple cases: `argmax`, comparison, type-cast, and mean are all you need. The pattern `(condition).float().mean()` for computing the fraction of elements satisfying a condition is idiomatic PyTorch and appears constantly in training loops, evaluation scripts, and unit tests throughout the course.

---

## (F1) total price indexing

```python
def total_price_indexing(prices: Tensor, items: Tensor) -> float:
    return prices[items].sum().item()
```

`prices[items]` is an example of integer array indexing (also called fancy indexing): instead of a single integer or slice, we index with a tensor of integers. PyTorch (like NumPy) interprets this as "gather the elements at these positions", returning a tensor of the same shape as `items` whose values are `prices[items[0]], prices[items[1]], ...`. This is fundamentally different from slice indexing — there is no constraint that the indices be contiguous or sorted, and the same index can appear multiple times (e.g., buying two of the same item).

Summing the gathered tensor with `.sum()` and extracting the Python scalar with `.item()` gives the total cost. This pattern — index into a lookup table, then aggregate — is one of the most common operations in deep learning: embedding lookups (indexing into a weight matrix with token IDs), gathering logits at target positions, and computing per-class statistics all follow exactly this structure. Understanding it at the simple 1D level here builds the intuition needed for the multi-dimensional `gather` operations in F2 and F3.

---

## (F2) gather 2D

```python
def gather_2d(matrix: Tensor, indexes: Tensor) -> Tensor:
    assert matrix.ndim == indexes.ndim
    assert indexes.shape[0] <= matrix.shape[0]
    out = matrix.gather(1, indexes)
    assert out.shape == indexes.shape
    return out
```

`torch.gather(input, dim, index)` generalises integer array indexing to multiple dimensions. For `dim=1`, the rule is `out[i][j] = input[i][index[i][j]]` — the row coordinate `i` is kept from the output position, while the column coordinate is replaced by whatever `index[i][j]` specifies. This means each row can select a *different* set of columns, which is impossible with standard slice notation. The `input` and `index` tensors must have the same number of dimensions, and the output always has the same shape as `index`.

The three asserts capture the shape contracts: `ndim` equality ensures gather won't silently broadcast across mismatched rank; the row-count check ensures every row in `index` has a corresponding row in `matrix`; and the output shape check confirms gather's contract. Writing these asserts is good practice because `gather` errors can otherwise manifest as wrong values rather than exceptions — particularly when `index` values are out of bounds, which PyTorch does not always check in non-debug mode. The main use of 2D gather in the course is selecting the logit corresponding to the true label for each item in a batch, which is exactly exercise H4.

---

## (F3) total price gather

```python
def total_price_gather(prices: Tensor, items: Tensor) -> float:
    return prices.gather(0, items).sum().item()
```

This reimplements F1 using `torch.gather` along `dim=0`. For a 1D tensor, `prices.gather(0, items)` is equivalent to `prices[items]` — it gathers elements at the positions specified by `items`. The reason to introduce gather here despite the simpler indexing solution is to build familiarity with the function's semantics before encountering it in higher-dimensional settings (like F2 and H4), where plain fancy indexing cannot express the same operation.

The conceptual distinction between F1 and F3 is important: fancy indexing `prices[items]` works for any number of dimensions but always indexes *all* remaining axes implicitly, whereas `gather` requires you to specify the dimension explicitly and gives you fine-grained control over which axis is being indexed. In 1D they are interchangeable, but in 2D+ they diverge. Many ARENA exercises — especially in the transformers chapter — use `gather` to select one element per row from a 2D score matrix, which requires the per-row flexibility that only `gather` provides.

---

## (G) integer array indexing

```python
def integer_array_indexing(matrix: Tensor, coords: Tensor) -> Tensor:
    return matrix[tuple(coords.T)]
```

`coords` has shape `(batch, n)` where each row is an n-dimensional coordinate into `matrix`. Transposing gives shape `(n, batch)` — one row per axis of `matrix`. Converting to a Python tuple and passing it as the index unpacks these rows so that `matrix[row_indices, col_indices, ...]` is applied simultaneously across all coordinates. PyTorch (following NumPy) interprets a tuple of index tensors as "select `matrix[row_indices[i], col_indices[i], ...]` for each `i`", returning a 1D result of shape `(batch,)`.

The `tuple(coords.T)` trick is the canonical NumPy/PyTorch idiom for batched multi-dimensional indexing when the coordinates are stored in a `(batch, ndim)` matrix. Without the transpose, the index would be interpreted differently (as indexing into the first axis only). This pattern generalises to any number of dimensions — the same code works for 2D, 3D, or higher-rank tensors — making it a powerful building block for operations like sampling random positions from feature maps or looking up values in a 3D voxel grid.

---

## (H1) batched logsumexp

```python
def batched_logsumexp(matrix: Tensor) -> Tensor:
    C = matrix.max(dim=-1).values
    exps = t.exp(matrix - einops.rearrange(C, "n -> n 1"))
    return C + t.log(t.sum(exps, dim=-1))
```

The naive formula `log(sum(exp(row)))` suffers from numerical overflow when any element is large (e.g., `exp(1000) = inf`) and from underflow when all elements are very negative (e.g., `exp(-1000) = 0`). The stable version subtracts the row maximum `C` before exponentiating: `log(sum(exp(x_i - C))) + C = log(sum(exp(x_i) / exp(C))) + C = log(sum(exp(x_i))) - C + C`. This is mathematically identical but numerically safe because after subtracting the maximum, the largest value in each row is 0 and all others are ≤ 0, so `exp` never overflows and the dominant term is always 1.

The `einops.rearrange(C, "n -> n 1")` inserts a size-1 column dimension so that `C` broadcasts correctly against the 2D `matrix` (shape `(batch, n)`). Without this, PyTorch would align `C` (shape `(batch,)`) with the last axis `n`, subtracting a different scalar from each column rather than from each row. This is a concrete, high-stakes instance of the `keepdim` / `unsqueeze` lesson from C1: dimension alignment matters, and failing to add the axis produces silently wrong results. Logsumexp is a fundamental building block for numerically stable softmax, log-softmax, and cross-entropy, all of which follow in H2–H4.

---

## (H2) batched softmax

```python
def batched_softmax(matrix: Tensor) -> Tensor:
    exp = matrix.exp()
    return exp / exp.sum(dim=-1, keepdim=True)
```

Softmax converts a vector of raw scores (logits) into a probability distribution: it exponentiates each element and normalises by the sum so that all outputs are positive and sum to 1. The `exp()` call is applied elementwise across the entire `(batch, n)` matrix, and `exp.sum(dim=-1, keepdim=True)` computes the normalising constant for each row independently, returning shape `(batch, 1)`. The `keepdim=True` is essential — without it the sum would have shape `(batch,)` which would broadcast along the wrong axis.

An important property verified by the tests is *translation invariance*: `softmax(x + c) == softmax(x)` for any constant `c`. This follows from the normalisation: adding `c` multiplies every `exp` by `e^c`, which cancels in the ratio. This property motivates the numerically stable version used in practice (subtract the max before exponentiating) and is the connection between the naive H2 implementation and the stable H1 logsumexp. For small logit values this naive version is fine; for large values (e.g., in attention with large `d_model`), the stable version is essential.

---

## (H3) batched logsoftmax

```python
def batched_logsoftmax(matrix: Tensor) -> Tensor:
    C = matrix.max(dim=1, keepdim=True).values
    return matrix - C - (matrix - C).exp().sum(dim=1, keepdim=True).log()
```

Log-softmax is more numerically stable than computing `softmax` then taking the log, because the log and exp partially cancel. Starting from the stable shift `x' = x - C`, log-softmax reduces to `x'_i - log(sum(exp(x'_j)))`, which is what this implementation computes. Both `C` (the row maximum) and the log-sum-of-exp use `keepdim=True` so that they broadcast correctly across the column axis when subtracted from `matrix`.

Log-softmax outputs are always ≤ 0 (since softmax outputs are ≤ 1), and the dominant (largest) value in each row is exactly 0 (the exponentiated maximum, after subtracting `C`, contributes 1 to the sum, and `log(1) = 0`). This function is used in H4 as the building block for cross-entropy loss, because `cross_entropy = -log(softmax(logit_at_true_class))` — taking the log-softmax first and then negating is more stable and efficient than computing softmax and then log separately. This is also what PyTorch's own `F.cross_entropy` does internally.

---

## (H4) batched cross entropy loss

```python
def batched_cross_entropy_loss(logits: Tensor, true_labels: Tensor) -> Tensor:
    logprobs = batched_logsoftmax(logits)
    indices = einops.rearrange(true_labels, "n -> n 1")
    pred_at_index = logprobs.gather(1, indices)
    return -einops.rearrange(pred_at_index, "n 1 -> n")
```

Cross-entropy loss for a single example is the negative log-probability assigned to the correct class. After converting logits to log-probabilities with `batched_logsoftmax`, we need to extract the log-probability at the true class index for each example in the batch. This is exactly the 2D gather operation from F2: `logprobs.gather(1, indices)` selects `logprobs[i, true_labels[i]]` for each row `i`, returning shape `(batch, 1)`. The `einops.rearrange` calls handle the bookkeeping of adding and removing the size-1 dimension that `gather` requires on `indices`.

The final negation converts log-probabilities to losses: a perfect prediction (log-prob = 0) gives loss 0, and increasingly wrong predictions give increasingly large positive losses. The tests include edge cases: `logits = [-inf, -inf, 0]` with true label 2 gives loss 0.0 (certain correct prediction); equal logits with true label 0 gives `log(3)` (uniform over 3 classes); and `-inf` logits at the true class give `inf` loss (the model assigned zero probability to the correct answer). These verify that the numerical handling from H3 propagates correctly through to the final loss.

---

## (I1) collect rows

```python
def collect_rows(matrix: Tensor, row_indexes: Tensor) -> Tensor:
    return matrix[row_indexes]
```

Indexing a 2D matrix with a 1D integer tensor selects entire rows. `matrix[row_indexes]` returns a new matrix of shape `(k, n)` where `k = len(row_indexes)`, and `out[i] = matrix[row_indexes[i]]`. Indices can be repeated (to duplicate a row) or appear in any order (to permute rows), neither of which is possible with standard slice notation. This is the simplest form of integer array indexing in 2D: providing a 1D index tensor for the first axis implicitly selects all columns via an implicit `:` on the remaining axes.

This operation is ubiquitous in deep learning: embedding lookups (selecting word vectors for a batch of token IDs), gathering predictions for a subset of examples, and dataset sub-sampling all reduce to this pattern. Recognising `matrix[index_tensor]` as a row selection is the foundation for understanding more complex gather operations. The assert `row_indexes.max() < matrix.shape[0]` is a good defensive habit — out-of-bounds integer indices are undefined behaviour in some backends and can produce garbage values without raising an error.

---

## (I2) collect columns

```python
def collect_columns(matrix: Tensor, column_indexes: Tensor) -> Tensor:
    return matrix[:, column_indexes]
```

Column selection uses the same integer-array indexing mechanism as I1, but applied to the second axis with an explicit `:` on the first: `matrix[:, column_indexes]` returns shape `(m, k)` where `out[:, i] = matrix[:, column_indexes[i]]`. The `:` means "all rows", while `column_indexes` selects which columns to include and in what order. As with row indexing, columns can be repeated or reordered arbitrarily.

Contrasting I1 and I2 makes the axis-selection logic of fancy indexing concrete: the position of the index tensor in the bracket determines which axis it selects along. `matrix[idx]` → rows; `matrix[:, idx]` → columns; `matrix[row_idx, col_idx]` → individual elements (as in exercise G). This generalises straightforwardly: `matrix[:, :, idx]` selects along the third axis of a 3D tensor, and so on. This axis-aware indexing is the building block for operations like selecting a specific attention head's output from a `(batch, heads, seq, d_head)` tensor — a pattern that appears constantly in the transformer interpretability chapters.**
