# Einops Image Exercises — Detailed Explanations

`arr` is a 4D NumPy array of shape `(6, 3, 150, 150)`: 6 digit images, each with 3 colour channels (RGB), 150 pixels tall and 150 pixels wide. The named axes used throughout are `b` (batch/image index), `c` (channel), `h` (height), `w` (width).

---

## (1) Column-stacking

```python
arr1 = einops.rearrange(arr, "b c h w -> c (b h) w")
```

This exercise stacks all 6 images into a single tall column. The key operation is `(b h)` on the output side, which tells einops to merge the batch dimension `b` and the height dimension `h` into one combined dimension. Concretely, image 0 occupies rows 0–149, image 1 occupies rows 150–299, and so on, all sharing the same width `w`. The channel dimension `c` stays at the front, so the output shape is `(3, 900, 150)`. No data is copied or lost — every pixel from every image appears exactly once in the output, just repositioned by treating the batch axis as a vertical extension of the spatial height axis.

---

## (2) Column-stacking and copying

```python
arr2 = einops.repeat(arr[0], "c h w -> c (2 h) w")
```

Here we take only the first image (`arr[0]`, shape `(3, 150, 150)`) and duplicate it vertically. `einops.repeat` is the right tool whenever you need to broadcast a tensor by *explicitly* replicating data. The pattern `(2 h)` in the output means "repeat the height dimension 2 times and concatenate the copies", producing an output of shape `(3, 300, 150)`. The first copy occupies rows 0–149 and the second occupies rows 150–299 — both are identical pixel-for-pixel. This is analogous to `np.tile` or `np.repeat`, but expressed in a readable named-axis notation that makes the intent immediately clear.

---

## (3) Row-stacking and double-copying

```python
arr3 = einops.repeat(arr[0:2], "b c h w -> c (b h) (2 w)")
```

This exercise combines two ideas from the previous ones: stacking multiple images vertically (`(b h)`) *and* copying each resulting image horizontally (`(2 w)`). We slice the first two images with `arr[0:2]` to get shape `(2, 3, 150, 150)`. The `(b h)` merge stacks image 0 above image 1, giving a height of 300. Then `(2 w)` repeats the entire width twice, so the final output is `(3, 300, 300)` — a square composed of the two digits side by side, with the whole composition duplicated to fill the right half. The batch dimension is consumed by the vertical stacking, while the width repetition is a pure copy with no merging of distinct images.

---

## (4) Stretching

```python
arr4 = einops.repeat(arr[0], "c h w -> c (h 2) w")
```

This stretches the first image to double its height by repeating each row twice. The notation `(h 2)` is the key: inside a `repeat` call, this means "for each position along `h`, output 2 copies consecutively". Row 0 appears at output rows 0 and 1, row 1 appears at rows 2 and 3, and so on — unlike exercise (2) where the *entire image* was duplicated as a block, here the duplication interleaves at the row level. The result has shape `(3, 300, 150)` and looks like the original digit squashed flat because every horizontal scanline is drawn twice. This pattern (`(dim factor)` inside `repeat`) is the standard einops idiom for nearest-neighbour upsampling along a single axis.

---

## (5) Split channels

```python
arr5 = einops.rearrange(arr[0], "c h w -> h (c w)")
```

This separates the three colour channels (R, G, B) of the first image and lays them out side by side as three greyscale panels. The merge `(c w)` on the output combines the channel axis with the width axis, so channel 0 occupies columns 0–149, channel 1 occupies columns 150–299, and channel 2 occupies columns 300–449. The height axis `h` is unchanged. Because the channel dimension is fully absorbed into the width, the result is 2D — shape `(150, 450)` — and the display utility therefore treats it as a monochrome image. This is a useful diagnostic technique: inspecting R, G, B contributions separately to understand colour balance or to debug image pre-processing pipelines.

---

## (6) Stack into rows & cols

```python
arr6 = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2)
```

This arranges all 6 images into a 2-row × 3-column grid. The factorisation `(b1 b2)` on the input side tells einops to split the flat batch dimension of size 6 into two factors: `b1=2` rows and `b2=3` columns (inferred automatically since 6 ÷ 2 = 3). On the output side, `(b1 h)` merges the row-of-grid index with image height to produce vertical stacking, and `(b2 w)` merges the column-of-grid index with image width to produce horizontal stacking. The final shape is `(3, 300, 450)`. This factorisation pattern — splitting a batch axis into a 2D grid index — is extremely common in visualisation code and is one of the clearest demonstrations of einops's expressive power over raw NumPy reshape calls.

---

## (7) Transpose

```python
arr7 = einops.rearrange(arr[1], "c h w -> c w h")
```

This flips the spatial dimensions of the second image, swapping height and width to produce a transposed (reflected along the main diagonal) version. The channel axis `c` is left in place; only `h` and `w` are exchanged in the output string. The result has shape `(3, 150, 150)` — identical to the input since this image is square — but every pixel that was at position `(c, i, j)` is now at `(c, j, i)`. In a non-square image this would visibly change the aspect ratio. Under the hood einops translates this directly to a `numpy.transpose` or `torch.permute` call, but the named notation makes the axis permutation explicit and self-documenting, avoiding the common mistake of passing the wrong axis indices.

---

## (8) Shrinking (max pooling)

```python
arr8 = einops.reduce(arr, "(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)", "max", h2=2, w2=2, b1=2)
```

This is the most complex operation: it simultaneously lays out all 6 images in a 2×3 grid *and* halves the resolution of each image using 2×2 max pooling. The batch axis is split into `(b1 b2)` exactly as in exercise (6). The height axis is split into `(h h2)` with `h2=2`, meaning each group of 2 consecutive rows is treated as a pooling window; similarly `(w w2)` groups columns in pairs. The `"max"` reduction then takes the brightest pixel from each 2×2 block, which is the standard max-pool operation used in CNNs to achieve spatial down-sampling while retaining the most prominent activations. Because `h2` and `w2` are consumed by the reduction and do not appear in the output, the spatial dimensions are halved. The final shape is `(3, 150, 225)` — a 2×3 grid of images each at 75×75 resolution. This single `reduce` call collapses four conceptually distinct steps (batch split, grid layout, spatial windowing, pooling) into one readable expression.
