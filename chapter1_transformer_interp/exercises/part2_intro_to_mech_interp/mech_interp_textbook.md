# Mechanistic Interpretability of Transformers

### A Textbook Treatment of TransformerLens and Induction Circuits

*Based on ARENA 3.0 Chapter 1.2 and Elhage et al., "A Mathematical Framework for Transformer Circuits" (2021)*

---

## Table of Contents

1. [What Is Mechanistic Interpretability?](#1-what-is-mechanistic-interpretability)
2. [The Transformer as a Residual Stream](#2-the-transformer-as-a-residual-stream)
3. [Weight Matrices: Parameters and Their Roles](#3-weight-matrices-parameters-and-their-roles)
4. [Attention Heads: Classical Notation and Circuit Notation](#4-attention-heads-classical-notation-and-circuit-notation)
5. [Full Circuits and the Privileged Basis](#5-full-circuits-and-the-privileged-basis)
6. [The One-Layer Model: A Complete Path Decomposition](#6-the-one-layer-model-a-complete-path-decomposition)
7. [Activations, Caching, and the Hook System](#7-activations-caching-and-the-hook-system)
8. [Logit Attribution and Ablation](#8-logit-attribution-and-ablation)
9. [Induction Heads](#9-induction-heads)
10. [Reverse-Engineering Circuits from Weights](#10-reverse-engineering-circuits-from-weights)
11. [The FactoredMatrix Class](#11-the-factoredmatrix-class)
12. [The OV Copying Circuit](#12-the-ov-copying-circuit)
13. [The QK Prev-Token Circuit](#13-the-qk-prev-token-circuit)
14. [K-Composition and the Full Induction Circuit](#14-k-composition-and-the-full-induction-circuit)
15. [Composition Scores and Virtual Weights](#15-composition-scores-and-virtual-weights)
16. [The Two-Layer Path Expansion](#16-the-two-layer-path-expansion)

---

## 1. What Is Mechanistic Interpretability?

### Plain Language Explanation

Most machine-learning research asks a forward question: "How do I train a model that performs well?" Mechanistic interpretability asks the reverse forensic question: "Given a model that already performs well, what algorithm did it learn, and where in its weights does that algorithm live?"

The analogy to computer science is decompiling binary code. You have a compiled executable — the trained model — and your job is to recover the source code, or at least a human-readable description of the algorithm it implements. Unlike decompiling software, we do not have access to the original "source code" that was optimised; training is a stochastic search process, not a compilation. But transformers turn out to have an enormous amount of exploitable mathematical structure, so the reverse-engineering is often tractable.

Why does this matter? Because modern AI systems make high-stakes decisions, and "it usually works on our benchmark" is not a satisfying explanation when the system occasionally produces harmful or unexpected outputs. If we could read the model's algorithm the way we read code, we could audit it, debug it, and — eventually — certify it. The ultimate vision of Neel Nanda and the Anthropic interpretability team is to be able to do for neural networks what reverse engineers do for malware: understand exactly what the program does and why.

We live in an era where language models can write code, translate languages, and reason through complex problems — yet no one fully understands *how* they do it. Mechanistic interpretability is the discipline of reverse-engineering trained neural networks: given a model that has already learned to solve a task, can we recover the algorithm it uses?

This is distinct from most machine learning research, which asks "how do we build better models?" Mechanistic interpretability asks "how does *this* model work, from its weights?" It is a forensic discipline.

The goal, as articulated by Neel Nanda, is ambitious: *"to take a trained model and reverse engineer the algorithms the model learned during training from its weights."* We want to be able to read a model the way a computer scientist reads assembly code — not just to run it, but to understand what every piece does.

Why transformers? Because they are the dominant architecture in modern AI, because they have a great deal of exploitable linear structure (as Elhage et al. 2021 emphasize), and because their weights are directly accessible for analysis. The core claim of the Mathematical Framework paper is that **transformers have an enormous amount of linear structure**, and that one can learn a great deal simply by breaking apart sums and multiplying together chains of matrices.

The **TransformerLens** library (Nanda, 2022) was built specifically for this purpose: to make it easy to load pre-trained models, access every internal activation, and intervene on the model's computation in controlled ways.

---

## 2. The Transformer as a Residual Stream

### Plain Language Explanation

Imagine a long assembly line, where a partly-assembled product moves from station to station and each station makes an incremental improvement. Crucially, each station does not discard what previous stations built — it only adds to it. The transformer's residual stream is exactly this assembly line. For each token in the input sequence, there is a vector that travels through the model from layer 0 to layer $L$, and every attention head and MLP layer *adds* something to that vector rather than replacing it.

This additive design has a profound consequence for interpretability: because the final output is a sum, it can be decomposed. If we want to know "how much did layer-2 head 4 contribute to predicting the word 'king'?", we can compute exactly that number. The decomposition is not an approximation — it is a mathematical identity that follows directly from the additive structure of residual connections. Without residual connections this decomposition fails entirely, which is one reason transformers are more interpretable than, for example, plain RNNs.

A second consequence is that all components share a common interface — the residual stream vector — and they communicate through it. Each component reads from the residual stream by projecting it through a weight matrix into a smaller subspace, does some computation in that small space, and writes the result back. This read-project-compute-write pattern repeats for every head and every MLP layer. The Mathematical Framework (Elhage et al., 2021) uses this pattern to decompose the transformer into independent circuits that can be analysed one at a time.

The single most important concept for mechanistic interpretability is the **residual stream**. Understanding it properly changes how you think about every other component.

### 2.1 Definition

In a standard transformer, each layer's output is *added* to its input rather than replacing it. After the embedding step, the input is a matrix $X_0 \in \mathbb{R}^{T \times d_\text{model}}$ (one row per token position). Each attention layer and MLP layer computes a *delta* — a matrix of the same shape — and adds it in:

$$X_\ell = X_{\ell-1} + \text{Attn}_\ell(X_{\ell-1}) + \text{MLP}_\ell(X_{\ell-1} + \text{Attn}_\ell(X_{\ell-1}))$$

The sequence of vectors $X_0, X_1, \ldots, X_L$ at a single token position is called the **residual stream** for that position. It is a $d_\text{model}$-dimensional vector that gets updated additively throughout the forward pass.

### 2.2 The Residual Stream as a Communication Bus

Elhage et al. (2021) articulate the key insight: *"All components of a transformer — the token embedding, attention heads, MLP layers, and unembedding — communicate with each other by reading and writing to different subspaces of the residual stream."*

This reframing is powerful. Rather than thinking of layer $\ell$ as transforming the representation, think of every component as:

1. **Reading** from the current residual stream vector by projecting it into a small subspace (via weight matrices like $W_Q$, $W_K$, $W_V$).
2. **Computing** something in that small subspace.
3. **Writing** the result back to the residual stream by projecting back up (via $W_O$ for attention, $W_\text{out}$ for MLPs).

Because all writes are additive and all reads are linear projections, the model has a great deal of linear structure that we can exploit. The superposition of contributions from different heads and layers is not a metaphor — it is literally a vector sum.

### 2.3 Notation

Throughout this text, we follow TransformerLens's convention that **vectors are row vectors** and **weight matrices multiply on the right**. So if $x \in \mathbb{R}^{d_\text{model}}$ is a residual stream vector and $W \in \mathbb{R}^{d_\text{model} \times d_\text{out}}$ is a weight matrix, then the projected output is $x W \in \mathbb{R}^{d_\text{out}}$.

This differs from the convention in many textbooks (including the Elhage et al. paper itself, which uses column vectors) but is consistent throughout TransformerLens and is the convention we use in all formulas below, noting when the paper differs.

---

## 3. Weight Matrices: Parameters and Their Roles

### Plain Language Explanation

Before doing any circuit analysis, we need to be clear about what the model's "wiring" actually is. In a transformer, the wiring is the weight matrices — the learned numerical arrays that determine how information flows. Think of them as the transistors and resistors on a printed circuit board: they are physical (or, in our case, mathematical) constants that do not change once the model is trained. Understanding each weight matrix's role is the first step towards reading the circuit.

The distinction between parameters and activations is crucial and easy to blur. A **parameter** is a number in the weight matrix: it is fixed after training, the same for every input, and can be printed on paper and studied. An **activation** is a number computed during one specific forward pass: it changes for every different input sentence, and conceptually ceases to exist once that forward pass is over. When someone says "head 0.7 attends to the previous token," they are describing a pattern in *activations* — which might vary somewhat with input. When they say "head 0.7 has a previous-token QK circuit," they are making a claim about *parameters* — which is stronger and more fundamental, because it holds for all inputs.

The embedding matrix $W_E$ and unembedding matrix $W_U$ deserve special attention because they are the model's only connection to human-interpretable concepts. $W_E$ maps integer token IDs (numbers like 2304 for the word "king") to floating-point vectors in $\mathbb{R}^{768}$. $W_U$ does the reverse at the output. Everything in between — the entire residual stream — lives in a 768-dimensional space where individual coordinates have no fixed meaning. This asymmetry is important: the input and output are interpretable (they are words), but the internals are not directly interpretable on their own. All circuit analysis must eventually connect back through $W_E$ and $W_U$ to speak in terms of tokens.

Before we can analyze circuits, we need to be precise about the weight matrices and what they do.

### 3.1 Parameters vs. Activations

**Parameters** are the learned weights and biases of the model. They are fixed after training, do not change with the input, and can be inspected directly. In TransformerLens, every parameter is accessible via `model.W_E`, `model.W_Q`, etc.

**Activations** are the intermediate computed values during a single forward pass. They depend on the input and conceptually exist only for the duration of that forward pass. Examples include the query vectors `q`, key vectors `k`, value vectors `v`, and the attention pattern matrix `A`. The hook system (Section 7) provides access to these during a forward pass.

Understanding this distinction is critical because a statement like "the attention pattern for head 0.7 is a previous-token pattern" is a claim about an *activation* (which varies with input), not a parameter. The circuit analysis we do in later sections makes claims about *parameters* (which are fixed), which is a stronger and more fundamental kind of understanding.

### 3.2 The Embedding and Unembedding

The **embedding matrix** $W_E \in \mathbb{R}^{d_\text{vocab} \times d_\text{model}}$ maps discrete token indices to continuous vectors. If $t$ is a one-hot vector of length $d_\text{vocab}$ representing a single token, then $t W_E \in \mathbb{R}^{d_\text{model}}$ is that token's embedding. In practice, TransformerLens represents this as a lookup: `model.W_E[token_id]`.

The **unembedding matrix** $W_U \in \mathbb{R}^{d_\text{model} \times d_\text{vocab}}$ is the inverse operation: it maps from the residual stream back to logits over the vocabulary. The final logits are $x_\text{final} W_U \in \mathbb{R}^{d_\text{vocab}}$, and the predicted next token is the argmax of these logits after softmax.

For GPT-2 Small, $d_\text{vocab} = 50257$, $d_\text{model} = 768$, so $W_E$ has shape $[50257 \times 768]$ and $W_U$ has shape $[768 \times 50257]$.

### 3.3 Positional Embeddings

Transformers have no inherent sense of token order, so positional information must be injected. The **positional embedding matrix** $W_\text{pos} \in \mathbb{R}^{n_\text{ctx} \times d_\text{model}}$ provides a separate $d_\text{model}$-dimensional vector for each position up to the context length $n_\text{ctx}$.

The toy two-layer model used throughout this chapter uses **Shortformer positional embeddings**: positional embeddings are added only to the inputs of the query and key projections, not to the values or the residual stream directly. This means $q = (x + p) W_Q$, $k = (x + p) W_K$, but $v = x W_V$, where $p$ is the positional embedding for that position. The practical effect is that the OV circuit (Section 4) cannot access positional information — it operates purely on token content. This design choice makes induction heads easier to train and easier to interpret.

### 3.4 Attention Head Weight Matrices

Each attention head $h$ in layer $\ell$ has four weight matrices:

| Matrix                                                       | Shape                             | Role                                              |
| ------------------------------------------------------------ | --------------------------------- | ------------------------------------------------- |
| $W_Q^h \in \mathbb{R}^{d_\text{model} \times d_\text{head}}$ | $[d_\text{model}, d_\text{head}]$ | Projects residual stream to query space           |
| $W_K^h \in \mathbb{R}^{d_\text{model} \times d_\text{head}}$ | $[d_\text{model}, d_\text{head}]$ | Projects residual stream to key space             |
| $W_V^h \in \mathbb{R}^{d_\text{model} \times d_\text{head}}$ | $[d_\text{model}, d_\text{head}]$ | Projects residual stream to value space           |
| $W_O^h \in \mathbb{R}^{d_\text{head} \times d_\text{model}}$ | $[d_\text{head}, d_\text{model}]$ | Projects from value space back to residual stream |

In TransformerLens, these are stored with an extra `head_index` axis: `model.W_Q` has shape `[n_layers, n_heads, d_model, d_head]`. This makes it possible to index a specific head with `model.W_Q[layer, head]`, without any reshaping.

For GPT-2 Small: $n_\text{layers} = 12$, $n_\text{heads} = 12$, $d_\text{model} = 768$, $d_\text{head} = 64$ (since $768 / 12 = 64$).

---

## 4. Attention Heads: Classical Notation and Circuit Notation

### Plain Language Explanation

Attention is the mechanism by which one token can "look at" other tokens in the sequence and gather information from them. The classical description uses three named vectors per token: a **query** ("what am I searching for?"), a **key** ("what do I advertise to other tokens?"), and a **value** ("what information will I share if I am chosen?"). The attention mechanism computes how well each query matches each key (producing a distribution of attention weights), then takes a weighted average of values according to those weights. This is essentially a differentiable lookup table: given a query, retrieve the most relevant values from the sequence.

A key insight from Elhage et al. (2021) is that the four weight matrices $W_Q$, $W_K$, $W_V$, $W_O$ do not act independently — they can be collapsed into just two interpretable matrices. The query and key matrices together define *where to attend* (a function of pairs of positions), and the value and output matrices together define *what to transfer* (a function of the content at the attended position). These two roles are largely independent, which is why the circuit view collapses Q and K into a single bilinear form $W_{QK}$, and V and O into a single linear map $W_{OV}$.

This circuit-view compression is not just notational convenience. It makes cross-head analysis tractable. To ask "does head 1.4 copy tokens?", you only need to look at $W_{OV}^{1.4}$, without caring about the particular factorization into $W_V$ and $W_O$. To ask "does head 0.7 attend to the previous position?", you only look at $W_{QK}^{0.7}$, without caring about the particular values of $W_Q$ and $W_K$. The decomposition peels away implementation details and reveals the head's function.

Attention is the core computation of a transformer. We present it in two complementary notations: the classical step-by-step view, and the compressed circuit notation of Elhage et al. (2021).

### 4.1 Classical QKV Notation

For a single attention head $h$ operating on a sequence of $T$ tokens with residual stream $X \in \mathbb{R}^{T \times d_\text{model}}$:

**Step 1 — Compute queries, keys, and values:**
$$Q = X W_Q^h \in \mathbb{R}^{T \times d_\text{head}}$$
$$K = X W_K^h \in \mathbb{R}^{T \times d_\text{head}}$$
$$V = X W_V^h \in \mathbb{R}^{T \times d_\text{head}}$$

Each row of $Q$ is the query for that token: "what am I looking for?" Each row of $K$ is the key: "what do I advertise to other tokens?" Each row of $V$ is the value: "what information will I share if attended to?"

**Step 2 — Compute the attention pattern:**
$$A^h = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_\text{head}}}\right) \in \mathbb{R}^{T \times T}$$

The $(i,j)$ entry $A^h_{i,j}$ is the probability that token $i$ attends to token $j$. The scaling by $1/\sqrt{d_\text{head}}$ prevents the dot products from becoming too large in high dimensions, which would push softmax into a near-deterministic regime. For causal (autoregressive) models, the upper triangle of the pre-softmax scores is masked to $-\infty$ so that tokens cannot attend to future positions.

**Step 3 — Compute the head output and write back:**
$$Z^h = A^h V \in \mathbb{R}^{T \times d_\text{head}}$$
$$\text{head\_out}^h = Z^h W_O^h \in \mathbb{R}^{T \times d_\text{model}}$$

The row $Z^h_i$ is a weighted average of the value vectors, with weights given by $A^h_{i,\cdot}$. Then $W_O^h$ projects this back to the full $d_\text{model}$ space. The final contribution of head $h$ to the residual stream is $\text{head\_out}^h$, which is added in.

The total output of all heads in a layer is the sum $\sum_h \text{head\_out}^h$, which is then added to the residual stream.

### 4.2 The Circuit View: W_QK and W_OV

The classical QKV notation reveals the computation step by step, but it obscures the function of each head. The Mathematical Framework (Elhage et al., 2021) compresses the four weight matrices into two **circuit matrices** that directly describe what the head does:

**The OV circuit** (output-value):
$$W_{OV}^h := W_V^h W_O^h \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$$

This is the combined "read-then-write" map. For any source token with residual stream vector $x_j$, the quantity $x_j W_{OV}^h$ is exactly what gets added to the destination position's residual stream, *if* the destination attends fully to that source. The two-step process $v = x_j W_V^h$, then $v W_O^h$ is collapsed into a single matrix product.

**The QK circuit** (query-key):
$$W_{QK}^h := W_Q^h (W_K^h)^T \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$$

This is the bilinear form that computes attention scores. For query position $i$ and key position $j$, the unnormalized attention score is:
$$\text{score}(i \to j) = x_i W_{QK}^h x_j^T = (x_i W_Q^h)(x_j W_K^h)^T = q_i \cdot k_j$$

Both $W_{OV}^h$ and $W_{QK}^h$ are $d_\text{model} \times d_\text{model}$ matrices. This is important: they operate entirely in residual stream space, making cross-head and cross-layer analysis tractable.

### 4.3 Convention Note: Paper vs. TransformerLens

The Elhage et al. paper uses **column-vector convention** (vectors are columns, matrices multiply on the left). Their definitions are:
$$W_{QK}^{h,\text{paper}} := (W_Q^h)^T W_K^h, \qquad W_{OV}^{h,\text{paper}} := W_O^h W_V^h$$

When you transpose every weight matrix (to convert from paper convention to TransformerLens row-vector convention), you get:

- $W_{QK}^{h,\text{TL}} = W_Q^h (W_K^h)^T$ — **the same matrix as the paper** (the bilinear form produces the same scalar)
- $W_{OV}^{h,\text{TL}} = W_V^h W_O^h$ — the **transpose** of the paper's $W_{OV}$

This is why the notebook notes: "the order of these matrices are slightly different from the Mathematical Frameworks paper — this is a consequence of the way TransformerLens stores its weight matrices." The key circuit identities remain the same; only the order of multiplication is reversed.

---

## 5. Full Circuits and the Privileged Basis

### Plain Language Explanation

Here is a core difficulty in interpreting neural networks: the numbers inside the model are not directly meaningful. Take a particular neuron — say, dimension 42 of the residual stream at layer 3. Is it encoding "the sentence is about food"? "the previous word was a verb"? "something to do with position 7"? The answer is: we cannot know just by looking at that number. The residual stream has no fixed coordinate system — it is just an abstract 768-dimensional space where the model stores information in whatever format it found convenient during training.

Mathematically, this is the statement that the residual stream has no privileged basis. You can apply any rotation $R$ (an orthogonal matrix satisfying $R R^T = I$) to the residual stream at every point in the model, and the model's input-output behaviour is completely unchanged. This is because every weight matrix that reads from or writes to the residual stream will absorb the rotation: if the residual stream is rotated by $R$, the weight matrix is effectively rotated by $R^{-1}$, and the composition cancels. No experiment on the model's outputs can distinguish between a model with residual stream $x$ and one with residual stream $xR$.

The only coordinates that are pinned down are the ones at the boundary: the embedding $W_E$ (which maps specific tokens to the residual stream) and the unembedding $W_U$ (which maps the residual stream to specific token logits). These define the "token basis" — the interpretable coordinate system. Any circuit that we claim to understand must be stated in terms of token-to-token relationships, connecting through $W_E$ and $W_U$. This is why the Mathematical Framework defines "full circuits" that compose the internal circuit matrices with $W_E$ on the left and $W_U$ on the right.

The circuit matrices $W_{OV}^h$ and $W_{QK}^h$ operate on residual stream vectors. But residual stream vectors are not directly interpretable: there is no "privileged basis" in residual stream space. The token vocabulary, on the other hand, is perfectly interpretable — each index corresponds to a specific string. Bridging from residual stream operations to vocabulary-level operations is the purpose of the **full circuits**.

### 5.1 The Privileged Basis Argument

Elhage et al. (2021) make a precise observation: you can insert any orthogonal rotation $R$ (satisfying $R R^T = I$) between every pair of components in the model — after each write to the residual stream, and before each read — without changing the model's output. This is because every component reads from the residual stream via a weight matrix ($W_Q$, $W_K$, $W_V$, $W_\text{in}$, $W_U$) and writes back via another weight matrix ($W_O$, $W_\text{out}$, $W_E$). If you replace $x \to xR$ everywhere, the weights absorb the rotation: $W \to R^T W$, and $R (R^T W) = W$. The computation is identical.

This means there is **no privileged basis** for the residual stream. A particular residual stream direction like "dimension 42" has no intrinsic meaning — it could be rotated away. The only basis that is privileged is the **token basis**: the one-hot vocabulary vectors that are mapped through $W_E$ (on the input side) and $W_U$ (on the output side). These are interpretable because they correspond to concrete strings.

Consequently, the only circuits that can be read off directly are end-to-end: from input tokens (or positions) to output logits.

### 5.2 The Full OV Circuit

Multiplying through both ends:
$$\text{Full OV}^h := W_E \, W_{OV}^h \, W_U \in \mathbb{R}^{d_\text{vocab} \times d_\text{vocab}}$$

Row $t$ of this matrix is the logit-vector change caused at the destination position when it fully attends to source token $t$. More precisely, if the destination token attends entirely to a source whose embedding is $t \, W_E$, then the logits shift by $(t \, W_E) W_{OV}^h W_U = t \cdot (\text{Full OV}^h)$.

The $(s, d)$ entry of the full OV circuit answers: "if the destination attends to source token $s$, by how much does it increase the logit for token $d$?" A large positive entry at $(t, t)$ means the head copies token $t$ to itself — a "copying head". A large off-diagonal entry at $(s, d)$ means attending to $s$ promotes predicting $d$ — which could represent translation, negation, syntactic completion, etc.

### 5.3 The Full QK Circuit

$$\text{Full QK}^h := W_E \, W_{QK}^h \, W_E^T \in \mathbb{R}^{d_\text{vocab} \times d_\text{vocab}}$$

Entry $(t_i, t_j)$ is the unnormalized attention score from a query token $t_i$ to a key token $t_j$, considering only the content-based (non-positional) component. Large values at $(t_i, t_j)$ mean "token $t_i$ tends to attend to token $t_j$".

There is also a positional version:
$$\text{Full Positional QK}^h := W_\text{pos} \, W_{QK}^h \, W_\text{pos}^T \in \mathbb{R}^{n_\text{ctx} \times n_\text{ctx}}$$

Entry $(i, j)$ of this matrix is the attention score contributed by the positional encodings at positions $i$ and $j$. For a previous-token head, this matrix should have large values just below the main diagonal (entry $(i,\, i-1)$ is large, meaning position $i$ attends to position $i-1$).

---

## 6. The One-Layer Model: A Complete Path Decomposition

### Plain Language Explanation

The one-layer attention-only model is the simplest transformer that has non-trivial behaviour, and the Mathematical Framework gives a completely closed-form analysis of it. This is a rare situation in deep learning: we can write down an exact algebraic formula for what the model computes, with no approximations. Understanding this formula thoroughly provides the foundation for understanding deeper models.

The key insight is that the model's prediction decomposes into two additive parts. The first part is the **direct path**: the model takes the current token, looks it up in the embedding table, projects directly to logit space via the unembedding, and makes a prediction. This prediction ignores everything else in the context — it is essentially asking "what word usually follows this word?", which is a bigram language model. This is the simplest possible language model, and it is always present in a transformer as one of the terms in the decomposition.

The second part is a sum of **attention head contributions**. Each attention head looks at the context, computes how much attention to pay to each previous position, and then contributes to the prediction based on what it attended to. Specifically, head $h$ contributes to the prediction at position $s$ by computing a weighted average of the full OV circuit evaluated at each context position $s'$: the weight is how much $s$ attends to $s'$ (from the attention pattern), and the value is what the full OV circuit says to predict when attending to the token at $s'$. This whole second part is also exact — no approximations.

The combined formula shows that a one-layer model implements an **ensemble of bigrams and skip-trigrams**: it predicts the next token based on the current token (bigram term) and based on selected earlier tokens (skip-trigram terms), where "selected" is determined by the learned attention patterns. This is significantly more powerful than a pure bigram model, but still limited: the model cannot do proper in-context learning, because each head computes attention based only on the raw embeddings (no cross-layer composition).

For an attention-only one-layer transformer (no MLP, no LayerNorm), the Mathematical Framework gives a complete closed-form expression for the output logits.

Let $T = (t_1, t_2, \ldots, t_n)$ be the input token sequence. The residual stream starts as:
$$x_s^{(0)} = t_s W_E + p_s$$
where $t_s$ is the one-hot vector for position $s$'s token and $p_s$ is its positional embedding. After the attention layer:
$$x_s^{(1)} = x_s^{(0)} + \sum_h \sum_{s'} A^h_{s,s'} \cdot x_{s'}^{(0)} W_{OV}^h$$

The final logits at position $s$ are:
$$\text{logits}[s] = x_s^{(1)} W_U = x_s^{(0)} W_U + \sum_h \sum_{s'} A^h_{s,s'} \cdot x_{s'}^{(0)} W_{OV}^h W_U$$

Substituting $x_{s'}^{(0)} = t_{s'} W_E + p_{s'}$ and noting that the attention pattern itself depends on $x^{(0)}$, we get a full decomposition into interpretable terms:

- **Direct path**: $t_s W_E W_U$ — the logit boost from the current token's embedding, bypassing all attention heads. This is essentially a bigram prior: tokens that frequently precede certain next tokens get those next-token logits boosted.
- **Head $h$ contribution**: $\sum_{s'} A^h_{s,s'} \cdot t_{s'} W_E W_{OV}^h W_U$ — for each head $h$, the weighted sum of full OV circuit rows, weighted by how much position $s$ attends to position $s'$.

The paper observes that a one-layer attention-only model effectively implements an **ensemble of skip-trigram and bigram models**: it can learn to predict the next token after "A [anything] B" even if that three-token sequence was rare in training, as long as the attention head learns to attend from the position after "B" to the token "A".

---

## 7. Activations, Caching, and the Hook System

### Plain Language Explanation

When debugging a computer program, you often insert print statements to observe intermediate values, or use a debugger that lets you step through the code and inspect the program's state at any point. TransformerLens provides exactly these capabilities for neural network analysis.

The `run_with_cache` method is the equivalent of running the program with logging turned on. It executes a single forward pass and saves every named intermediate value — every query matrix, every attention pattern, every residual stream state at every layer — in a dictionary called the cache. You can then look up any of these values after the forward pass completes. The naming convention follows the PyTorch module structure: `cache["pattern", 0]` gives you the $12 \times T \times T$ attention pattern tensor (one $T \times T$ matrix per head) for layer 0.

Hooks are more powerful than caching because they allow you to not just observe but also intervene. A hook is a function you attach to a named activation point in the model; it runs every time that activation is computed, receives the activation tensor, and can return a modified tensor. This is how we do causal interventions: we set an activation to zero (ablation), replace it with a fixed value, or add noise to it, and observe how the model's output changes. If ablating head 0.7 causes the model to stop attending to the previous token, that is direct causal evidence that head 0.7 implements the previous-token behavior.

### 7.1 Accessing Activations with run_with_cache

To inspect a model's internal states for a given input, TransformerLens provides the method:

```python
logits, cache = model.run_with_cache(tokens)
```

The `cache` object stores every named intermediate activation from the forward pass. Activations are named by their location in the computational graph, e.g.:

- `cache["embed"]` — token embeddings, shape `[batch, seq, d_model]`
- `cache["pos_embed"]` — positional embeddings
- `cache["q", 0]` — query vectors at layer 0, shape `[batch, seq, n_heads, d_head]`
- `cache["k", 0]` — key vectors at layer 0
- `cache["v", 0]` — value vectors at layer 0
- `cache["pattern", 0]` — attention pattern at layer 0, shape `[batch, n_heads, seq, seq]`
- `cache["z", 0]` — weighted sum of values at layer 0, shape `[batch, seq, n_heads, d_head]`
- `cache["resid_pre", 1]` — residual stream before layer 1
- `cache["resid_post", 0]` — residual stream after layer 0

You can verify the attention calculation manually: `cache["pattern", 0]` should equal `softmax(cache["q", 0] @ cache["k", 0].T / sqrt(d_head))` (with appropriate einsum and masking).

### 7.2 The Hook System

Hooks allow you to intercept and optionally modify any activation during a forward pass, without re-running the whole model. In TransformerLens, every named activation is wrapped in a `HookPoint` module. You can attach a Python function to any `HookPoint`, and it will be called with `(activation, hook)` every time that activation is computed.

A hook function has the signature:

```python
def hook_fn(activation: Tensor, hook: HookPoint) -> Tensor:
    # read, modify, or replace `activation`
    return activation  # must return a tensor of the same shape
```

To run a model with hooks:

```python
loss = model.run_with_hooks(
    tokens,
    return_type="loss",
    fwd_hooks=[("blocks.1.attn.hook_pattern", hook_fn)]
)
```

The hook name `"blocks.1.attn.hook_pattern"` is the full PyTorch module path to the attention pattern at layer 1. You can use `utils.get_act_name("pattern", 1)` as a shorthand.

Hooks are the primary tool for **causal interventions**: by modifying an activation, you test whether a particular component is responsible for a particular behavior. If you set an activation to zero and the model's performance on the task drops, that component was doing something important.

### 7.3 Ablation

**Ablation** is the practice of silencing a component to measure its causal importance. The two main variants are:

- **Zero ablation**: replace the component's output with the zero vector. This is the simplest form and asks "what happens if this component does nothing?"
- **Mean ablation**: replace the component's output with its average over a batch of inputs. This asks "what happens if this component contributes only its average behavior, with no input-specific information?"

Zero ablation is easier to interpret but can be misleading: a head outputting zero might be out-of-distribution for downstream components that have learned to expect some baseline contribution. Mean ablation is more conservative.

For example, to zero-ablate head $h$ in layer 1:

```python
def head_zero_ablation_hook(z, hook, head_index_to_ablate):
    z[:, :, head_index_to_ablate, :] = 0.0
    return z

model.run_with_hooks(
    tokens,
    fwd_hooks=[("blocks.1.attn.hook_z",
                partial(head_zero_ablation_hook, head_index_to_ablate=4))]
)
```

---

## 8. Logit Attribution and Ablation

### Plain Language Explanation

Logit attribution is the application of the additive decomposition principle to the question "which components made the right prediction?" Since the final logit vector is the sum of contributions from every head and the direct path, the logit for a specific token (say, "sat") is also a sum of contributions. Each head's contribution is a single real number: the dot product of that head's output (written to the residual stream) with the column of the unembedding matrix for the target token.

This number — the logit attribution score — has a clean interpretation. A positive score means the head's output raised the model's probability for the correct token. A negative score means the head hurt the prediction. A score near zero means the head was irrelevant to this particular prediction. By computing these scores for all heads on a given input, you get a "scoreboard" that tells you which parts of the model were responsible for the model's answer.

The logit attribution decomposes what would otherwise be an opaque model output into an auditable sum. If the model predicts the wrong word, you can pinpoint which head was most responsible for the error. If a head in layer 0 is consistently responsible for predicting "sat" after "cat", that is strong evidence that the head has learned the specific word association "cat → sat." This bridges from model-level behavior ("the model predicted correctly") to component-level behavior ("head 1.4 contributed +2.3 logits to the correct token on this input").

### 8.1 Decomposing Logits into Component Contributions

Because the residual stream is additive, the final logits can be decomposed into contributions from each component. The key identity is:

Let $o^{\ell,h}$ denote the output written to the residual stream by head $h$ in layer $\ell$. Because the residual stream is a sum of all contributions:

$$\text{logits} = x_\text{final} W_U = \left(x_0 + \sum_{\ell, h} o^{\ell,h}\right) W_U = x_0 W_U + \sum_{\ell, h} o^{\ell,h} W_U$$

where $x_0 = t W_E + p$ is the initial residual stream (embedding + positional embedding). Each term $o^{\ell,h} W_U$ is the direct contribution of head $(\ell, h)$ to the output logits. This decomposition is exact, not an approximation.

For a given input sequence and a target next-token $t^*$, the **logit attribution** for head $(\ell, h)$ is:
$$\delta_{\ell,h} = o^{\ell,h}_{[s]} \cdot W_U^\text{correct}$$

where $W_U^\text{correct} = W_U[:, t^*]$ is the column of $W_U$ corresponding to the correct next token. This scalar tells you how much head $(\ell, h)$ contributed to predicting the correct next token, and can be positive (helpful) or negative (harmful).

### 8.2 Logit Lens

A closely related idea is the **logit lens**: at any intermediate position in the residual stream (e.g., after layer $\ell$), you can project the current residual stream through $W_U$ to see what token it would predict if the model stopped there. This allows you to track how the model's prediction evolves layer by layer.

---

## 9. Induction Heads

### Plain Language Explanation

Induction heads are one of the most important discovered circuits in large language models, and understanding them thoroughly pays off across all of mechanistic interpretability. The core behaviour is this: if the sequence "the cat sat" appeared earlier in the context, and "cat" appears again later, an induction head can recognise that "cat" was previously followed by "sat" and predict "sat" will come again. Crucially, this pattern-matching happens dynamically from the context window, not from memorised training data — the model can generalise to completely novel word pairs it has never seen during training.

This in-context learning ability is qualitatively different from what a one-layer model can do. A one-layer model can only ask "what token usually follows this token?" (a bigram) or "given that I see A and B in context, predict C" (a skip-trigram). Neither of these captures "the same pattern that appeared 50 tokens ago should be repeated now." That requires reading two separated occurrences of the same token and connecting them — an operation that requires at least two layers.

The circuit that implements induction is beautifully simple: it requires exactly two attention heads in two consecutive layers. The first head (the "previous-token head") always attends to the position immediately to the left, and writes "what token was to my left?" into the residual stream. The second head (the "induction head") reads the modified residual stream, and uses the "what was to my left?" information as part of its key computation. This allows it to find positions where the token to the left was the current query token — which is exactly the positions that occurred after the previous occurrence of the current token. The induction head then copies what followed at those positions into the current prediction.

### 9.1 What Are Induction Heads?

**Induction heads** are a specific type of attention head, typically found in two-layer transformers, that implement a form of **in-context learning**: given a sequence containing the pattern $A \, B \ldots A$, an induction head attending from the second $A$ will attend back to the first $B$ (the token that followed the first $A$) and copy its information forward, predicting that $B$ will follow the second $A$ as well.

The formal definition (Elhage et al., 2021): an induction head is one whose attention pattern, on a repeating sequence $[t_1, t_2, \ldots, t_n, t_1, t_2, \ldots, t_n]$, exhibits a distinctive **diagonal stripe offset by $n-1$** positions. Token at position $n + i$ attends to position $i + 1$ (the token immediately following the previous occurrence of the current token).

Induction heads are significant for several reasons:

- They emerge **suddenly** during training, at a distinct phase transition (around 2B–4B tokens for language models in the relevant size range). The loss curve shows a visible bump when they form.
- They are responsible for the majority of **in-context learning**: the ability to use far-back context to improve next-token prediction. This is something that older architectures like RNNs and LSTMs struggle to do.
- The same circuit appears to be reused for more sophisticated tasks, including few-shot translation and reasoning patterns — suggesting that induction is a fundamental primitive that transformers learn to generalize.

### 9.2 The Induction Circuit: Step by Step

The induction head does not operate alone. It requires two heads in two layers working in concert:

**Layer 0: The Previous-Token Head (e.g., head 0.7)**

The previous-token head attends from each position to the position immediately before it. Its QK circuit, using the positional component:
$$W_\text{pos} \, W_{QK}^{0.7} \, W_\text{pos}^T \in \mathbb{R}^{n_\text{ctx} \times n_\text{ctx}}$$
has large values just below the main diagonal: entry $(i,\, i-1)$ dominates, so position $i$ attends to position $i-1$.

Its OV circuit copies the attended token's information back to the current position's residual stream. So after layer 0, the residual stream at position $i$ now also contains information about token $t_{i-1}$ (the previous token), written there by head 0.7.

**Layer 1: The Induction Head (e.g., head 1.4), via K-Composition**

The induction head's keys are not computed from the raw token embedding (like a normal head) but from the *output of head 0.7* that was written to the residual stream. This is **K-composition**: head 1.4's key input is partly determined by head 0.7's output.

As a result, at position $j$, the key vector effectively encodes: "the token before me is $t_{j-1}$." The query at the second occurrence of $t_i$ looks for positions where the previous token was $t_i$, finding the key at position $i+1$ (right after the first $t_i$). The induction head attends there and copies the value $t_{i+1}$ (the token that followed the first $t_i$) into the prediction.

### 9.3 Why a Single Layer Is Insufficient

A single-layer attention head computes its attention scores purely from the content of the query token and the content of the key token — there is no mechanism to incorporate "what is next to token $j$?" into the attention score.

The induction head needs to attend to position $j$ based on the *value of the token at position $j+1$*. That is beyond what a single layer's QK circuit can express. The previous-token head in layer 0 is what makes this possible: it encodes into position $j$'s residual stream the information "my right neighbor is token $t$", allowing the layer-1 head to attend based on this composed key.

**Composition between layers exponentially increases the expressivity of the circuit.** Where a one-layer model can only implement bigram and skip-trigram patterns, a two-layer model can implement induction, which is a form of one-shot in-context learning.

### 9.4 Detecting Induction Heads Empirically

To detect induction heads without circuit analysis, run the model on a repeated random sequence:
$$[t_1, t_2, \ldots, t_n, t_1, t_2, \ldots, t_n]$$

For the first half of the sequence, the model has no information to use for in-context prediction. For the second half, an induction head can perfectly predict each token (since it appeared $n$ positions earlier). The per-token loss on the second half should be dramatically lower than on the first half — this is direct evidence that the model has learned induction.

Visually, the attention pattern of an induction head on this sequence is unmistakable: a bright diagonal stripe offset from the main diagonal by $n-1$ positions.

---

## 10. Reverse-Engineering Circuits from Weights

### Plain Language Explanation

There is a fundamental difference between two types of evidence in mechanistic interpretability. The first type is observational: you run the model on an input, look at the attention patterns, and say "on this input, head 0.7 attended mostly to the previous position." This is useful evidence, but it is input-dependent — the same head might behave differently on different inputs. The second type is mechanistic: you multiply the weight matrices together, examine the resulting matrix directly, and say "head 0.7's QK circuit is structured so that it always attends to the previous position." This is a claim about the model's fixed parameters, which holds for all inputs.

The difference is like the difference between observing that a specific function call returns 42, versus reading the function's source code and seeing it always returns 42. The second is a stronger claim and requires no input to verify. In the context of the induction circuit, we can go from "the model often gets the second occurrence right in a repeated sequence" (observational) to "the product $W_E W_{OV}^{0.7} W_{QK}^{1.4} W_E^T$ has the structure you'd expect for induction" (mechanistic). The second tells us not just that induction happens, but which specific weights are responsible and precisely how they implement it.

Weight-based analysis also decomposes complex circuits into sub-circuits that can be verified independently. Instead of asking "does the induction circuit work?", we ask four separate questions: does head 0.7's OV circuit copy token identity? does head 0.7's QK circuit attend to the previous position? does head 1.4's OV circuit copy the attended token into logits? does head 1.4's QK circuit (via K-composition) attend to the right place? Each question can be answered by examining a specific product of weight matrices.

### 10.1 Two Approaches to Interpretability

There are two complementary ways to understand what a model is doing:

1. **Activation-based analysis**: run the model on specific inputs, observe the activations (attention patterns, intermediate residual stream values), and infer the mechanism from these observations. This is empirical and input-dependent.

2. **Weight-based analysis**: inspect the weight matrices directly, multiply them together to form circuit matrices, and derive the mechanism from those. This is input-independent and constitutes a stronger claim — you are describing what the model *always* does, not just what it does on a particular input.

Weight-based analysis is the "gold standard" of mechanistic interpretability, per Elhage et al. (2021). When we can derive the behavior of a circuit algebraically from the weights, we have a complete mechanistic explanation.

### 10.2 The Induction Circuit's Four Sub-Circuits

For the toy two-layer attention-only model with induction heads at 1.4 and 1.10 and a previous-token head at 0.7, the full induction circuit breaks into four sub-circuits:

1. **OV circuit of head 0.7** (prev-token head): determines *what* the prev-token head copies from position $j$ to position $j+1$. We want this to copy the token identity of $t_j$.

2. **QK circuit of head 0.7**: determines *where* the prev-token head attends. We want it to attend from position $i$ to position $i-1$.

3. **OV circuit of head 1.4** (induction head): determines *what* the induction head copies to the destination. We want it to copy token identity (i.e., to be a "copying" OV circuit).

4. **QK circuit of head 1.4**: determines *where* the induction head attends, subject to **K-composition** with head 0.7. This is the most complex sub-circuit.

---

## 11. The FactoredMatrix Class

### Plain Language Explanation

The full OV circuit $W_E W_{OV}^h W_U$ connects every token in the vocabulary to every other token. For GPT-2's vocabulary of 50,257 tokens, this is a 50,257 × 50,257 matrix — roughly 2.5 billion floating-point numbers, or about 10 GB in 32-bit precision. You cannot store this on a GPU, and even if you could, forming it would require multiplying a 50,257 × 768 matrix by a 768 × 50,257 matrix, which is a $O(50257^2 \times 768)$ computation. This is completely infeasible.

But here is the key insight: the circuit is not really a 50,257 × 50,257 matrix. It is the product of thin matrices with 64-dimensional bottlenecks in the middle. The sequence is: $W_E \in \mathbb{R}^{50257 \times 768}$, then $W_V^h \in \mathbb{R}^{768 \times 64}$, then $W_O^h \in \mathbb{R}^{64 \times 768}$, then $W_U \in \mathbb{R}^{768 \times 50257}$. The information must flow through a 64-dimensional bottleneck. This means the matrix $W_E W_{OV}^h W_U$, while formally 50,257 × 50,257, has rank at most 64 — it has at most 64 non-zero singular values, out of a theoretical maximum of 50,257. We can work with the thin factors instead of the full matrix.

The `FactoredMatrix` class in TransformerLens makes this practical. It stores a matrix $M = AB$ as the pair $(A, B)$ and implements all the operations you care about — Frobenius norm, trace, eigenvalues, SVD, top-$k$ entries, matrix multiplication with other matrices — by working with $A$ and $B$ directly, never forming $M$. The eigenvalue trick is particularly elegant: the non-zero eigenvalues of $AB$ are the same as the non-zero eigenvalues of $BA$, and $BA$ is a 64 × 64 matrix whose eigenvalues are trivial to compute.

### 11.1 The Problem: Giant Matrices

The full OV circuit $W_E W_{OV}^h W_U$ is a $d_\text{vocab} \times d_\text{vocab} = 50278 \times 50278$ matrix — approximately 2.5 billion floating-point numbers. Even in 16-bit precision, this requires about 5 GB of memory. Materializing it directly is infeasible.

However, the circuit is not really that large. It is a product of thin matrices:
$$(W_E W_V^h)(W_O^h W_U) \in \mathbb{R}^{50278 \times 64} \cdot \mathbb{R}^{64 \times 50278}$$

The actual information content of the circuit lives in the $d_\text{head} = 64$-dimensional bottleneck. The large matrix is low-rank: it has at most 64 non-zero singular values, in a space of 50,278 dimensions.

Similarly, the full QK circuit can be written as:
$$(W_E W_Q^h)(W_E W_K^h)^T \in \mathbb{R}^{50278 \times 64} \cdot \mathbb{R}^{64 \times 50278}$$

### 11.2 FactoredMatrix: Operations on the Factored Form

The `FactoredMatrix` class in TransformerLens stores a matrix $M = AB$ as its two factors $A$ and $B$, and implements key matrix operations directly on the factors without ever materializing $M$:

**Frobenius norm**: The Frobenius norm satisfies $\|M\|_F^2 = \sum_{i,j} M_{ij}^2 = \sum_k \sigma_k(M)^2$ (sum of squared singular values). Using the SVD of the smaller factors, this can be computed in $O(d \cdot d_\text{head}^2)$ rather than $O(d^2 \cdot d_\text{head})$.

**Trace**: $\text{Tr}(AB) = \sum_i \sum_j A_{ij} B_{ji}$, computable in $O(d \cdot d_\text{head})$.

**Eigenvalues**: The non-zero eigenvalues of $M = AB$ equal the non-zero eigenvalues of $BA$ (a much smaller $d_\text{head} \times d_\text{head}$ matrix). Proof: if $AB v = \lambda v$ with $\lambda \neq 0$, then $BA(Bv) = B(ABv) = \lambda (Bv)$, so $Bv$ is an eigenvector of $BA$ with the same eigenvalue. This is exact for all non-zero eigenvalues.

**SVD**: Computing the SVD of $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times m}$ takes $O(mk^2)$ each. Combining their SVDs gives the full SVD of $M$ cheaply.

### 11.3 Why Eigenvalues Matter: The Copying Score

For an OV circuit, the eigenvalues directly diagnose whether the head copies tokens:

- **Positive eigenvalues**: attending to source token $t$ *increases* the logit for $t$ at the destination — the head is copying.
- **Negative eigenvalues**: attending to $t$ *decreases* the logit for $t$ — anti-copying.
- **Near-zero eigenvalues**: the head moves information but does not preserve token identity.

The **copying score** (Elhage et al., 2021) is the diagonal of the full OV circuit $W_E W_{OV}^h W_U$: entry $(t, t)$ measures how much attending to token $t$ increases the logit for token $t$.

Computing this from the eigenvalues of $W_O^h W_V^h$ (a $d_\text{head} \times d_\text{head}$ matrix) is efficient:
$$\lambda_i(W_V^h W_O^h) = \lambda_i(W_O^h W_V^h), \quad i = 1, \ldots, d_\text{head}$$

---

## 12. The OV Copying Circuit

### Plain Language Explanation

For the induction circuit to correctly predict the continuation of a repeated sequence, the induction head (1.4) must do two things: find the right position to attend to (handled by the QK circuit and K-composition, discussed in Section 14), and once there, copy the token at that position into the model's prediction (handled by the OV circuit discussed here). This section focuses on the second task.

"Copying" means attending to token X and boosting the logit for token X. In vocabulary space, the full OV circuit $W_E W_{OV}^h W_U$ should look approximately like an identity matrix: if you attend to token "sat", the logit for "sat" should increase. If the circuit were a perfect identity, every entry along the diagonal would be equal and every off-diagonal entry would be zero. In practice, the circuit is approximately diagonal but not exactly, because the model also generalises (e.g., attending to "sat" might also slightly boost "sits" or "sitted" as related words).

To verify this empirically, we compute the top-1 accuracy: for each of the 50,000+ vocabulary tokens, feed it as the source through the full OV circuit and ask "is the source token itself the highest-logit prediction?" A value close to 100% confirms a strong copying head. We can also compute this more efficiently by looking at the eigenvalues of the small matrix $W_O^h W_V^h$ (64 × 64), which must be mostly positive for the head to be a copier. Both methods converge on the same conclusion for head 1.4.

### 12.1 Analysis

For the induction circuit to work, the induction head must copy the token it attends to into the logits. The OV circuit of head 1.4 (and similarly 1.10) should therefore be a near-identity map from source token to predicted next token.

To verify this, we compute the full OV circuit:
$$W_E W_{OV}^{1.4} W_U = W_E W_V^{1.4} W_O^{1.4} W_U$$

and examine whether it is approximately a diagonal matrix (in the vocabulary basis) — which would mean each source token maps primarily to itself.

The `top_1_accuracy` diagnostic asks: for what fraction of source tokens $t$ does the full OV circuit predict $t$ as the top-1 logit? A value near 1.0 confirms that head 1.4 is a strong copying head.

### 12.2 Effective OV Circuit

Since heads 1.4 and 1.10 both play the role of induction head and both attend to the same positions (in the induction pattern), their OV circuits combine linearly. The **effective OV circuit** is:
$$W_E (W_{OV}^{1.4} + W_{OV}^{1.10}) W_U = W_E (W_V^{1.4} W_O^{1.4} + W_V^{1.10} W_O^{1.10}) W_U$$

This additive combination works because the final logits from both heads are summed in the residual stream. The effective circuit typically has higher copying accuracy than either head alone, suggesting the two induction heads are jointly responsible for the copying behavior.

---

## 13. The QK Prev-Token Circuit

### Plain Language Explanation

To understand the previous-token head (0.7), we need to verify that it attends from position $i$ to position $i-1$ systematically, across all inputs. We cannot just look at attention patterns on a few examples — those are activations that depend on the input. Instead, we want a weight-based argument.

The key observation is that (for the Shortformer model used in ARENA exercises) positional embeddings are added to the Q and K projections but not to V. This means the attention score between position $i$ (query) and position $j$ (key) includes a positional contribution of $p_i W_{QK}^{0.7} p_j^T$, where $p_i$ and $p_j$ are the positional embeddings for positions $i$ and $j$. This positional contribution is a matrix in $\mathbb{R}^{n_\text{ctx} \times n_\text{ctx}}$ whose $(i,j)$ entry tells us how much position $i$ wants to attend to position $j$ based purely on their relative positions.

If head 0.7 is a previous-token head, this positional QK circuit should look like a lower-diagonal stripe: entry $(i, i-1)$ should be the largest in each row $i$ after softmax. When we compute $W_\text{pos} W_{QK}^{0.7} W_\text{pos}^T$ and apply softmax, we should see that stripe. This is a direct weight-based proof — it holds for all inputs — and it is much stronger than showing that the attention pattern looks like a stripe on a few test sentences.

### 13.1 Analysis

Head 0.7 is the **previous-token head**: it should attend from position $i$ to position $i-1$ for all $i$. The content-based component of attention is negligible here; the pattern is driven entirely by positional information.

We verify this by computing the full positional QK circuit:
$$W_\text{pos} \, W_{QK}^{0.7} \, W_\text{pos}^T \in \mathbb{R}^{n_\text{ctx} \times n_\text{ctx}}$$

After applying softmax (with the same scaling as in the actual attention computation), this should give a matrix where entry $(i, i-1)$ is close to 1 and all other entries are close to 0. Plotting this matrix produces a clear sub-diagonal stripe.

The off-diagonal structure confirms that the head's attention pattern is determined almost entirely by relative position, not token content — which is exactly what a previous-token head should do.

---

## 14. K-Composition and the Full Induction Circuit

### Plain Language Explanation

K-composition is the mechanism that makes the induction head more than the sum of its parts. To see why it is necessary, consider what head 1.4's attention score computation would look like without any composition. In that case, the key at position $j$ would be $x_j W_K^{1.4}$ where $x_j$ is just the raw embedding of token $t_j$. The attention score from position $i$ (query) to position $j$ (key) would then be determined entirely by the *content of tokens $t_i$ and $t_j$*. The head would learn which pairs of tokens tend to attend to each other, but it could not learn anything about what surrounds position $j$.

K-composition adds a second term to position $j$'s key: the contribution from head 0.7's output $\Delta x_j^{0.7} = \sum_{s'} A_{j,s'}^{0.7} \cdot x_{s'}^{(0)} W_{OV}^{0.7}$. Since head 0.7 attends primarily to position $j-1$, this term is approximately $x_{j-1}^{(0)} W_{OV}^{0.7}$ — the residual stream of the token to the left of $j$, after being processed through head 0.7's OV circuit. The key at position $j$ now encodes not only "what is $t_j$?" but also "what is the token at position $j-1$?"

This is the breakthrough: the query at position $i$ (the second occurrence of $t_i$) can now search for keys where "the token to the left is $t_i$". The position that satisfies this condition is $i+1$ from the first occurrence — right after where $t_i$ first appeared. The attention head finds that position and attends to it, then copies its token (the one that followed the first $t_i$) into the current prediction. The composed key has turned a "what is at this position?" question into a "what came right after token $t_i$ last time?" question.

### 14.1 What K-Composition Means

K-composition occurs when a later head's **key** computation is influenced by the **output** of an earlier head. Formally, if head $h_1$ writes output $\Delta x^{h_1}$ to the residual stream, and head $h_2$ computes its keys as $k = (x + \Delta x^{h_1}) W_K^{h_2}$, then head $h_2$'s attention pattern is partly determined by $\Delta x^{h_1} W_K^{h_2}$.

In the induction circuit, the K-composition is between head 0.7 (which writes prev-token information) and head 1.4 (whose keys read from that written information). The key at position $j$ in head 1.4 is approximately:
$$k_j^{1.4} \approx x_j W_K^{1.4} + (\Delta x^{0.7}_j) W_K^{1.4} = x_j W_K^{1.4} + (x_{j-1} W_{OV}^{0.7}) W_K^{1.4}$$

The second term encodes: "what is the K-projection of the previous token's representation?"

### 14.2 The Full K-Composition Circuit

The K-composition contribution to the attention score at head 1.4 from query position $i$ to key position $j$ is:
$$x_i W_{OV}^{0.7} W_{QK}^{1.4} x_j^T$$

Expanding through the full vocabulary:
$$W_E \, W_{OV}^{0.7} \, W_{QK}^{1.4} \, W_E^T \in \mathbb{R}^{d_\text{vocab} \times d_\text{vocab}}$$

This is the **full K-composition circuit**. Entry $(A, B)$ represents: "how much does head 1.4 attend to source token $A$, for a query position that attended to token $B$ in head 0.7?"

For the induction circuit, large values should appear at $(B, A)$ — meaning: "attend to $B$ (= the token that followed the previous $A$) when the current query token recently saw $A$ via head 0.7." This is exactly the induction behavior.

There is also a positional K-composition term:
$$W_\text{pos} \, W_{OV}^{0.7} \, W_{QK}^{1.4} \, W_\text{pos}^T \in \mathbb{R}^{n_\text{ctx} \times n_\text{ctx}}$$

which encodes how the positional offset information propagates from head 0.7 to head 1.4's attention pattern.

### 14.3 Why QK and OV Act Semi-Independently

The attention pattern (determined by the QK circuit) and the information moved (determined by the OV circuit) are largely independent. The QK circuit decides *where* to look; the OV circuit decides *what to copy* once you've decided where to look.

This semi-independence means the hard part of implementing induction is in the QK side (requiring K-composition across two layers), while the OV side is relatively simple (a near-identity copying map).

---

## 15. Composition Scores and Virtual Weights

### Plain Language Explanation

When we claim that head 0.7 "composes" with head 1.4, what we mean precisely is that head 0.7's output is relevant to head 1.4's input. But in a 768-dimensional space, each head writes into a 64-dimensional subspace and reads from a 64-dimensional subspace. For head 0.7's output to be relevant to head 1.4's input, those two subspaces must overlap. If the two heads had been initialised randomly and independently, there is no reason for them to share a subspace — in high dimensions, two random 64-dimensional subspaces in $\mathbb{R}^{768}$ are nearly orthogonal with high probability.

The composition score is a quantitative measure of subspace overlap. Concretely, we compute $W_A W_B$ (the product of head 0.7's OV circuit with head 1.4's QK circuit) and measure its Frobenius norm, normalised by the individual norms of $W_A$ and $W_B$. If $W_A$'s column space (what it writes) is orthogonal to $W_B$'s row space (what it reads), then $W_A W_B \approx 0$ regardless of how large $W_A$ and $W_B$ individually are. If the column space of $W_A$ aligns with the row space of $W_B$, then $W_A W_B$ will have large singular values and a large Frobenius norm.

The random baseline for this score is approximately $1/\sqrt{d_\text{model}} \approx 0.036$ for GPT-2 Small. Any pair of heads with a composition score significantly above this baseline has carved out a shared subspace during training, which is strong evidence that they are part of a functional circuit. By computing these scores for all pairs of heads in the two-layer model, we can build a map of the circuit structure: which pairs are wired together, and through which mechanism (K, Q, or V).

### 15.1 The Core Idea

The residual stream is $d_\text{model} = 768$-dimensional. Each attention head reads from and writes to a much smaller $d_\text{head} = 64$-dimensional subspace. By default, any two heads occupy essentially orthogonal subspaces in 768 dimensions — two random $64$-dimensional subspaces in $\mathbb{R}^{768}$ are nearly perpendicular with high probability.

If two heads are deliberately composing, they must have coordinated to share a subspace: the earlier head's output subspace overlaps with the later head's input subspace. This overlap is measurable, and measuring it is the purpose of **composition scores** (Elhage et al., 2021, §"Residual Comms").

### 15.2 The Composition Score

Elhage et al. define the composition score between two weight matrices $W_A$ and $W_B$ as:
$$C(W_A, W_B) = \frac{\|W_A W_B\|_F}{\|W_A\|_F \cdot \|W_B\|_F}$$

where $\|\cdot\|_F$ is the Frobenius norm: $\|M\|_F = \sqrt{\sum_{i,j} M_{ij}^2} = \sqrt{\sum_k \sigma_k^2}$.

**Baseline**: for two random low-rank matrices in $\mathbb{R}^{d \times k}$ with $k \ll d$, the expected composition score is approximately $1/\sqrt{d_\text{model}}$. For our model, $1/\sqrt{768} \approx 0.036$. A score meaningfully above this indicates genuine composition.

### 15.3 The Three Types of Composition

The paper identifies three types of composition, depending on *which input wire* of the later head $h_B$ is fed by the earlier head $h_A$:

| Composition       | What composes                    | $W_A$          | $W_B$                                       |
| ----------------- | -------------------------------- | -------------- | ------------------------------------------- |
| **Q-composition** | $h_A$'s output → $h_B$'s queries | $W_{OV}^{h_A}$ | $W_{QK}^{h_B}$                              |
| **K-composition** | $h_A$'s output → $h_B$'s keys    | $W_{OV}^{h_A}$ | $(W_{QK}^{h_B})^T = W_K^{h_B}(W_Q^{h_B})^T$ |
| **V-composition** | $h_A$'s output → $h_B$'s values  | $W_{OV}^{h_A}$ | $W_{OV}^{h_B}$                              |

For the induction circuit, we expect **K-composition** between head 0.7 and heads 1.4 / 1.10 to show high scores. The other heads should have composition scores near the random baseline.

### 15.4 Virtual Weights and Virtual Attention Heads

When composition occurs, Elhage et al. (2021) define the resulting **virtual weight** as the product $W_A W_B$. For K-composition between heads $h_A$ and $h_B$:
$$\text{Virtual K-comp circuit} = W_{OV}^{h_A} \cdot (W_{QK}^{h_B})^T$$

This is a single $d_\text{model} \times d_\text{model}$ matrix. The composition of two linear functions is another linear function — two heads composing via K-composition are equivalent to a single "virtual" head with a larger effective parameter count.

The full token-level virtual circuit:
$$W_E \, W_{OV}^{h_A} \, W_{QK}^{h_B} \, W_E^T \in \mathbb{R}^{d_\text{vocab} \times d_\text{vocab}}$$

is exactly the K-composition circuit analyzed in Section 14. This is the bridge between composition scores (a *diagnostic* for whether two heads are composing) and the full circuit analysis (a *mechanistic* explanation of what they are composing to do).

---

## 16. The Two-Layer Path Expansion

### Plain Language Explanation

The two-layer path expansion is the exact mathematical answer to the question "what does a two-layer transformer compute?" It is a complete expansion of the model's logits into a sum of interpretable terms. Each term corresponds to one "path" through the network: a sequence of choices about which components to pass through from the input token embedding to the output logit.

The first three types of terms are straightforward: the direct path (embedding → logit, no attention), the layer-0 head contributions (each head's direct addition to the residual stream, then unembedded), and the layer-1 head contributions reading only from the initial residual stream (same structure as a one-layer model). These three types together give you everything a one-layer model can do, plus whatever layer-1 heads add on top when reading the original embeddings.

The fourth type — composition terms — is qualitatively new. These arise when a layer-1 head's query, key, or value computation is influenced by a layer-0 head's output. Each composition term introduces a cross-layer product of weight matrices. In the K-composition case, the term involves $W_{OV}^{h_0} W_{QK}^{h_1}$: the layer-0 head's OV circuit feeding into the layer-1 head's QK circuit. This is what enables the induction circuit. The path expansion shows explicitly that induction is not a property of either layer alone — it is a property of the cross-layer composition term. Without two layers, the term simply does not exist.

This decomposition has a practical use: by computing how much of the model's output comes from each type of term on a specific task, you can identify which aspects of the task require composition (and thus require depth) and which aspects are handled by the single-layer terms. For language modelling, empirical studies of GPT-2 find that the composition terms contribute significantly to the model's performance on long-range dependencies, exactly as the theory predicts.

### 16.1 Full Decomposition

For a two-layer attention-only model, the Mathematical Framework gives a complete expansion of the logits as a sum over **paths** through the network. Each path corresponds to a sequence of operations — a subset of attention heads — and the sum of all paths gives the exact output.

Let $x_0 = t W_E + p$ be the initial residual stream. The final logits are:

$$\text{logits} = \underbrace{x_0 W_U}_{\text{(1) direct path}} + \underbrace{\sum_{h \in L_0} \sum_{s'} A^h_{s,s'}\, x_{s'}^{(0)} W_{OV}^h W_U}_{\text{(2) } L_0 \text{ head contributions}} + T_{L_1}$$

where $T_{L_1}$ collects the layer-1 terms:

$$T_{L_1} = \underbrace{\sum_{h \in L_1} \sum_{s'} A^h_{s,s'}\, x_{s'}^{(0)} W_{OV}^h W_U}_{\text{(3) } L_1 \text{ heads, direct residual stream}} + \underbrace{\text{K/Q/V-composition terms}}_{\text{(4) } L_1 \text{ heads reading } L_0 \text{ outputs}}$$

The composition terms arise from the fact that the residual stream fed to $L_1$ includes the outputs of $L_0$ heads. There are three types of composition terms (K, Q, V), one for each way an $L_1$ head can be influenced by $L_0$ outputs.

### 16.2 Why This Decomposition Matters

The path expansion shows that a two-layer model is not just "two times more powerful" than a one-layer model — it has qualitatively different expressive power. The composition terms enable computations that are impossible in one layer:

- **Induction** (K-composition): predicting token $B$ following $A$ based on having seen $A \to B$ earlier in context.
- **Indirect object identification** (Q-composition): predicting that the indirect object should be different from the subject, by attending from a query determined by one head to a key determined by another.
- **Value-composition** chains: cascading information through the OV circuits of multiple heads.

### 16.3 Limitations

The Mathematical Framework applies cleanly to **attention-only models**. MLPs are much harder to analyze: they introduce nonlinearity (the activation function, usually GELU or ReLU), which breaks the linear structure that makes circuits tractable. The paper explicitly identifies this as "a major weakness of our work."

Nevertheless, the circuit analysis framework — W_OV, W_QK, full circuits, composition scores, virtual weights — provides the conceptual toolkit that researchers have since applied (with modification) to full models including GPT-2, Pythia, and GPT-4.

---

## Summary: Key Definitions

| Term                  | Definition                                                                                                       |
| --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Residual stream**   | The $d_\text{model}$-dimensional vector at each token position that accumulates all component outputs additively |
| **W_QK**              | $W_Q (W_K)^T \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$; bilinear form for attention scores          |
| **W_OV**              | $W_V W_O \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$; combined "read-then-write" map                  |
| **Full OV circuit**   | $W_E W_{OV}^h W_U \in \mathbb{R}^{d_\text{vocab} \times d_\text{vocab}}$; maps source tokens to logit changes    |
| **Full QK circuit**   | $W_E W_{QK}^h W_E^T \in \mathbb{R}^{d_\text{vocab} \times d_\text{vocab}}$; maps token pairs to attention scores |
| **Privileged basis**  | Token embedding/unembedding space; the only basis that has semantic meaning                                      |
| **Direct path**       | Contribution $x_0 W_U$ to logits from the embedding, bypassing all attention heads                               |
| **Induction head**    | Head that attends to "token after previous occurrence of current token"                                          |
| **K-composition**     | When a later head's keys are partly determined by an earlier head's OV output                                    |
| **Composition score** | $\|W_A W_B\|_F / (\|W_A\|_F \|W_B\|_F)$; measures subspace overlap between heads                                 |
| **Virtual weight**    | $W_A W_B$; the effective single-step matrix for two composing heads                                              |
| **FactoredMatrix**    | Class for low-rank products $M = AB$; avoids materializing large matrices                                        |
| **Copying score**     | Diagonal of $W_E W_{OV}^h W_U$; measures how much a head copies tokens to themselves                             |
| **Ablation**          | Zeroing or mean-replacing a component's output to measure its causal importance                                  |

---

## References

- Elhage, N., Nanda, N., Olah, C., et al. (2021). *A Mathematical Framework for Transformer Circuits*. Transformer Circuits Thread. https://transformer-circuits.pub/2021/framework/index.html

- Nanda, N. (2022). *TransformerLens*. GitHub. https://github.com/TransformerLensOrg/TransformerLens

- Nanda, N. & Callum McDougall (2023). *ARENA 3.0: Mechanistic Interpretability*. https://arena-chapter1-transformer-interp.streamlit.app/

- Olah, C., et al. (2020). *Zoom In: An Introduction to Circuits*. Distill. https://distill.pub/2020/circuits/zoom-in/

- Elhage, N., et al. (2022). *In-Context Learning and Induction Heads*. Transformer Circuits Thread. https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
