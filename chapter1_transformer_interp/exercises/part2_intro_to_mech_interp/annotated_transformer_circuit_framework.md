# The Annotated Transformer — Circuit Framework Edition

*A section-by-section re-reading of Vaswani et al. (2017) and the Harvard NLP Annotated Transformer (Rush, 2018/2022), with every architectural component annotated using the notation and results of Elhage et al., "A Mathematical Framework for Transformer Circuits" (2021).*

*Goal: after reading this document you should be able to move fluently between the original "Attention is All You Need" equations, the Annotated Transformer's PyTorch implementation, and the circuit-analysis notation — seeing each as a different lens on the same underlying computation.*

---

## Table of Contents

1. [Background: Why Transformers, Why Circuits?](#1-background)
2. [Model Architecture at a Glance](#2-model-architecture-at-a-glance)
3. [The Residual Stream: A Circuit Framework Reframing](#3-the-residual-stream)
4. [Embeddings: The Privileged Basis](#4-embeddings-the-privileged-basis)
5. [Positional Encoding](#5-positional-encoding)
6. [Scaled Dot-Product Attention](#6-scaled-dot-product-attention)
7. [Multi-Head Attention](#7-multi-head-attention)
8. [The OV and QK Circuit Decomposition](#8-the-ov-and-qk-circuit-decomposition)
9. [Full Circuits: From Vocabulary to Vocabulary](#9-full-circuits-from-vocabulary-to-vocabulary)
10. [Position-wise Feed-Forward Networks](#10-position-wise-feed-forward-networks)
11. [Layer Normalization](#11-layer-normalization)
12. [Encoder–Decoder Attention and Cross-Attention](#12-encoderdecoder-attention-and-cross-attention)
13. [The Path Expansion: Logits as a Sum of Paths](#13-the-path-expansion)
14. [Zero-Layer Transformers: Bigram Statistics](#14-zero-layer-transformers)
15. [One-Layer Transformers: Skip-Trigrams and the OV Circuit](#15-one-layer-transformers)
16. [Two-Layer Transformers: Composition and Virtual Weights](#16-two-layer-transformers)
17. [Induction Heads](#17-induction-heads)
18. [Composition Scores](#18-composition-scores)
19. [Training: Adam, Warmup, Label Smoothing](#19-training)
20. [Summary: Notation Rosetta Stone](#20-summary-notation-rosetta-stone)

---

## 1. Background

### Plain Language Explanation

This document bridges two distinct research traditions. The first is the architecture engineering tradition, represented by Vaswani et al. (2017) "Attention Is All You Need" and the Harvard NLP Annotated Transformer by Alexander Rush. This tradition asks: how do we design a model that learns from data efficiently? It gives us the architecture — the blueprint of a transformer — and the training procedures. The second is the mechanistic interpretability tradition, represented by Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits." This tradition asks: given a trained model, what algorithm did it learn? These are fundamentally different questions, and combining the two perspectives gives a much richer understanding than either alone.

The Annotated Transformer is already a pedagogical achievement: it takes the original Vaswani et al. paper, which is written in mathematics, and walks through a clean PyTorch implementation line by line, making every equation into runnable code. But it does not ask why the components work or what they compute in a principled sense — it is descriptive, not explanatory. The Mathematical Framework fills this gap by deriving algebraic formulae for the model's computations directly from its weight matrices.

The key insight that makes the Mathematical Framework possible is that transformers have an enormous amount of **linear structure**. Residual connections mean outputs are sums. Attention reads and writes through linear projections. The composition of linear maps is another linear map. This linearity can be exploited to derive exact, input-free formulae for what each component does — something that is simply not possible in models with pervasive nonlinearities. While MLPs (and to a lesser extent LayerNorm) break this perfect linearity, the attention-only part of the model is analytically tractable and provides deep insight into what transformers are actually computing.

### 1.1 The Annotated Transformer perspective

Vaswani et al. (2017) introduced the Transformer to replace recurrent networks for sequence transduction. The key motivation was parallelism: RNNs process tokens sequentially, creating long dependency chains that are slow to train and hard to optimise. Self-attention computes all pairwise token interactions in one matrix multiplication, making full use of modern GPU hardware.

The Annotated Transformer (Rush, 2018; updated 2022) is a line-by-line implementation of that paper in PyTorch, making every equation concrete. It covers the encoder–decoder stack used for machine translation, along with the training schedule, label smoothing, and the beam-search decoding used in the original experiments.

### 1.2 The Mathematical Framework perspective

Elhage et al. (2021) approach the same architecture from a different angle: given a *trained* transformer, what algorithms did it learn? Their key observation is that transformers contain an **enormous amount of linear structure**, and that this structure can be exploited to derive exact algebraic descriptions of the model's behavior from its weights.

The framework applies most cleanly to **decoder-only, attention-only models** (no MLPs, no encoder) — precisely the architecture used in modern language models (GPT-2, GPT-3, Pythia, Llama). The encoder–decoder of Vaswani et al. is more complex, but every attention mechanism in it obeys the same circuit equations we derive below.

---

## 2. Model Architecture at a Glance

### Plain Language Explanation

The original Transformer from Vaswani et al. was designed for machine translation: you feed it a sentence in French, and it produces a sentence in English. The architecture splits this into two halves. The **encoder** reads the input (French) sentence and builds a rich representation of each word in context. The **decoder** generates the output (English) sentence one word at a time, attending both to the already-generated English words (via self-attention) and to the encoder's representation of the French input (via cross-attention).

Each half is made of $N$ identical layers stacked on top of each other. In the encoder, each layer has two sub-components: a self-attention block (where French words look at each other) and a feed-forward network. In the decoder, each layer has three sub-components: a causal self-attention block (English words look at previously generated English words), a cross-attention block (English words look at the French encoder output), and a feed-forward network.

The most important design decision from an interpretability standpoint is the **residual connection**: every sub-component's output is added to its input, not used to replace it. This means the final output of the model is literally a sum of contributions from every sub-component in every layer. You can ask "how much did this particular feed-forward layer contribute to the final logit for 'cat'?" and get an exact numerical answer. This additive structure is what makes the circuit framework possible.

### 2.1 Annotated Transformer view

The original Transformer is an **encoder–decoder** with $N$ identical layers in each stack. Each encoder layer has:

1. A multi-head self-attention sublayer
2. A position-wise feed-forward sublayer

Each decoder layer adds a third sublayer: multi-head cross-attention over the encoder output. Every sublayer wraps its computation with a **residual connection** and **layer normalisation**:

$$\text{Sublayer}(x) = \text{LayerNorm}(x + f(x))$$

where $f$ is the sublayer's function (attention or FFN). The model dimension is $d_\text{model}$ throughout; Vaswani et al. use $d_\text{model} = 512$ for the base model and $d_\text{model} = 1024$ for the large model.

### 2.2 Circuit Framework annotation

The **residual connection** is the single most important architectural choice for interpretability. Because every sublayer adds its output rather than replacing the input, the final output is literally a sum:

$$x_\text{final} = x_\text{embed} + \sum_{\text{layers}} \Delta_\text{attn} + \sum_{\text{layers}} \Delta_\text{FFN}$$

This means the transformer's output logits can be decomposed exactly into contributions from each component — a fact the Mathematical Framework exploits throughout. Without residual connections this decomposition fails entirely.

---

## 3. The Residual Stream

### Plain Language Explanation

Think of the residual stream as a shared whiteboard in a collaborative workspace. At the start of processing a token, someone writes the initial information about that token on the whiteboard (the token embedding). Then a series of experts — the attention heads and feed-forward layers — each look at the whiteboard, think about what they can add, and write their contribution. Crucially, no one erases what was previously written — every contribution is additive. At the end, a reader (the unembedding matrix) reads the entire whiteboard and produces the final prediction.

This whiteboard metaphor captures several important properties. First, information persists: something written at layer 0 is still on the whiteboard at layer 11, unless a later layer overwrites its subspace (but even then, the original contribution is simply cancelled out by an equal and opposite contribution, not erased). Second, all experts have access to everything: a layer-2 head can read information written by a layer-1 head, because both wrote to the same whiteboard. This is how composition between layers works — layer-2 components are responding not just to the original token, but to whatever layer-1 already wrote.

The key mathematical subtlety is that the whiteboard has no labelled sections. There is no "emotion region" or "syntactic role region" on the whiteboard — the 768 dimensions are arbitrary floating-point numbers, and different components might use the same dimension for completely different purposes depending on the input. This is why you cannot interpret individual residual stream dimensions directly: the whiteboard is written in a private coordinate system that the model chose during training, not in one that is meaningful to a human observer. The only interpretable "labels" on the whiteboard are at the boundaries: the token embedding and unembedding.

### 3.1 Definition

At each token position $s$, there is a $d_\text{model}$-dimensional vector called the **residual stream** $x_s$. It begins as the sum of the token embedding and positional encoding, and is updated additively by every attention head and FFN layer it passes through.

Let $x_s^{(\ell)}$ denote the residual stream at position $s$ after layer $\ell$. The update rule for an attention-only model is:

$$x_s^{(\ell)} = x_s^{(\ell-1)} + \sum_{h=1}^{H} \text{Attn}^{\ell,h}(x^{(\ell-1)})_s$$

### 3.2 The communication bus metaphor

Elhage et al. (2021) describe the residual stream as the **communication channel** of the transformer:

> "All components of a transformer — the token embedding, attention heads, MLP layers, and unembedding — communicate with each other by reading and writing to different subspaces of the residual stream."

Every component follows the same pattern:

- **Read**: project $x_s$ through a weight matrix into a small subspace (e.g. $x_s W_Q$ projects to query space of dimension $d_\text{head}$).
- **Compute**: do something in that small space (dot product, nonlinearity, etc.).
- **Write**: project the result back to $d_\text{model}$ and add it to $x_s$.

Because every read and write is linear (projections), and because writes are additive, the entire model has a great deal of structure that can be analyzed algebraically.

### 3.3 No privileged basis in the residual stream

A key insight: the residual stream does not have an interpretable coordinate system. You can insert any orthogonal rotation $R$ ($R R^T = I_d$) between every write and every read without changing the model's behavior, because weight matrices will absorb the rotation:

$$x W \to (xR)(R^T W) = xW$$

This means "residual stream dimension 42" carries no intrinsic meaning. The **only** interpretable bases are at the boundaries:

- **Input side**: the token embedding $W_E$, which maps discrete vocabulary indices to $\mathbb{R}^{d_\text{model}}$.
- **Output side**: the unembedding $W_U$, which maps $\mathbb{R}^{d_\text{model}}$ to logits over the vocabulary.

Any analysis of the model's behavior must ultimately be stated in terms of end-to-end circuits connecting vocabulary tokens to logits — not intermediate residual stream values.

---

## 4. Embeddings: The Privileged Basis

### Plain Language Explanation

Before a transformer can process text, it must convert discrete tokens (integers like 2304, representing the word "king") into continuous vectors that can be manipulated by matrix multiplications. This conversion is the job of the embedding matrix $W_E$. Think of it as a large lookup table: row 2304 of $W_E$ is the 768-dimensional vector representation of "king." When the model is trained, these row vectors are learned alongside all other parameters, so that words with similar meanings end up with similar vectors — a property known as the distributional hypothesis.

The unembedding matrix $W_U$ does the reverse at the output end. After all the attention and feed-forward processing, the residual stream at each position is a 768-dimensional vector. The unembedding maps this to a 50,000-dimensional vector of logits — one per vocabulary token. The softmax of these logits gives the probability distribution over the next token. In many models (including the original Transformer), $W_U = W_E^T$: the same matrix does both jobs, just in opposite directions. This weight tying is a form of regularisation that forces the model to use a consistent representation throughout.

From a circuit analysis perspective, $W_E$ and $W_U$ are the only places where the model interfaces with human-interpretable concepts. Everything inside the model — the residual stream, the query/key/value projections, the MLP hidden layer — lives in abstract numerical space. But $W_E$ and $W_U$ are indexed by vocabulary tokens, which are concrete strings. This makes them the privileged endpoints for all circuit analysis: any claim about what the model is computing must eventually be expressed as a relationship between input tokens (via $W_E$) and output token predictions (via $W_U$).

### 4.1 Annotated Transformer view

Both the encoder and decoder begin with a learned **token embedding**. The Annotated Transformer calls this `Embeddings`, a standard `nn.Embedding` table. In the original paper, the same weight matrix is used for:

1. The source token embedding.
2. The target token embedding.
3. The pre-softmax linear projection (unembedding).

All three are tied and scaled by $\sqrt{d_\text{model}}$. For language modelling (decoder-only) one typically has separate $W_E$ and $W_U$.

### 4.2 Circuit Framework annotation

**Embedding matrix** $W_E \in \mathbb{R}^{d_\text{vocab} \times d_\text{model}}$:
If $t$ is a one-hot token vector of length $d_\text{vocab}$, the embedding is $t W_E \in \mathbb{R}^{d_\text{model}}$.

**Unembedding matrix** $W_U \in \mathbb{R}^{d_\text{model} \times d_\text{vocab}}$:
The final logits at position $s$ are $x_s^{(L)} W_U \in \mathbb{R}^{d_\text{vocab}}$. When weight tying is used, $W_U = W_E^T$ (up to the $\sqrt{d_\text{model}}$ scaling).

**Privileged basis**: $W_E$ and $W_U$ define the only interpretable coordinate system in the model. The matrix product $W_E W_U \in \mathbb{R}^{d_\text{vocab} \times d_\text{vocab}}$ is the **direct path** (or **residual path**): what the model predicts from the embedding alone, without any attention computation. In a zero-layer model this is the entire model.

For GPT-2 style models: $d_\text{vocab} \approx 50257$, $d_\text{model} = 768$ (small), $1024$ (medium), $1280$ (large), $1600$ (XL).

---

## 5. Positional Encoding

### Plain Language Explanation

Self-attention has a fundamental symmetry: if you permute the tokens in the input sequence, the attention mechanism treats the permuted version identically (it just computes the same pairwise interactions in a different order). This means a pure self-attention model cannot tell whether "cat sat mat" and "mat cat sat" are different orderings — they would produce the same output. But word order is obviously crucial for meaning, so position information must be injected explicitly.

Vaswani et al. solve this with sinusoidal positional encodings: for each position $\text{pos}$ and each embedding dimension $i$, a fixed sinusoidal value is added to the embedding. The sinusoidal formula is chosen so that the relative position between two tokens can be expressed as a linear function of their encodings — a property that makes it easier for attention heads to compute relative positions. Later models (BERT, GPT-2, LLaMA) use learned positional embeddings instead: a matrix $W_\text{pos}$ where each row is the learned vector for a particular position, trained end-to-end alongside everything else.

An important variant used in some toy models is "Shortformer-style" positional embeddings, where position information is added only to the query and key inputs, not to the value or the residual stream. This creates a clean separation: the QK circuit (which determines *where* to attend) has access to position, but the OV circuit (which determines *what* to transfer) is purely content-based. This separation makes the model more interpretable, because you can analyse the routing and the content-copying independently.

### 5.1 Annotated Transformer view

Since self-attention is permutation-equivariant (it treats all positions identically), Vaswani et al. inject position information via **sinusoidal positional encodings** added to the embeddings before the first layer:

$$\text{PE}(\text{pos}, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_\text{model}}}\right)$$

$$\text{PE}(\text{pos}, 2i+1) = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_\text{model}}}\right)$$

where $\text{pos}$ is the token position and $i$ indexes dimensions. The encoding has the property that relative positions can be expressed as linear functions of the encoding at any single position. Later models (BERT, GPT-2) replace these with **learned positional embeddings** — a matrix $W_\text{pos} \in \mathbb{R}^{n_\text{ctx} \times d_\text{model}}$ that is trained rather than fixed.

### 5.2 Circuit Framework annotation

In the standard architecture (Vaswani et al., GPT-2), positional encodings are added to the residual stream at the very start:
$$x_s^{(0)} = t_s W_E + p_s$$
where $p_s = W_\text{pos}[s]$ is the $s$-th row of the positional embedding matrix.

Because $p_s$ is in the residual stream, it flows into every attention head's Q, K, and V computations equally.

**Shortformer / separated positional embeddings**: The toy models used in the ARENA exercises use a different scheme where positional embeddings are added only to the Q and K inputs, not to V or the residual stream:
$$q_s = (x_s + p_s) W_Q, \quad k_s = (x_s + p_s) W_K, \quad v_s = x_s W_V$$

This means the **OV circuit is positionally blind**: it operates purely on token content. The **QK circuit** has access to position. This factored design makes induction heads easier to train and easier to interpret, because positional and content-based attention can be cleanly separated.

**Full positional QK circuit**: When positional embeddings are in the residual stream (standard case), the QK circuit for head $h$ decomposes into a token-content component and a positional component:
$$\text{score}(i \to j) = x_i W_{QK}^h x_j^T = \underbrace{(t_i W_E) W_{QK}^h (t_j W_E)^T}_{\text{content}} + \underbrace{p_i W_{QK}^h p_j^T}_{\text{position}} + \text{cross terms}$$

The purely positional component $W_\text{pos} W_{QK}^h W_\text{pos}^T \in \mathbb{R}^{n_\text{ctx} \times n_\text{ctx}}$ tells you which positions attend to which positions based on location alone — and is what reveals the "previous-token" structure of heads like 0.7.

---

## 6. Scaled Dot-Product Attention

### Plain Language Explanation

Attention is a soft, differentiable lookup operation. The idea is to retrieve information from a set of source positions, weighted by how relevant each source is to the current destination. In the classical formulation, each position provides two vectors: a key (what it has to offer) and a value (the actual information it will share). The destination provides a query (what it is looking for). The attention mechanism computes the similarity between the query and every key — producing a score for each source — normalises these scores into a probability distribution with softmax, and then forms a weighted average of the values.

The scaling factor $1/\sqrt{d_k}$ is a numerical stability trick. In high dimensions, the dot product of two random unit vectors has expected value 0 but standard deviation $\sqrt{d_k}$ — so dot products grow as the dimension grows. Without scaling, the softmax would operate on scores with variance $d_k$, which pushes it toward extremely peaked (near-one-hot) distributions where gradient flow is poor. Dividing by $\sqrt{d_k}$ brings the variance back to $O(1)$, keeping the softmax in a reasonable operating regime.

The causal mask is essential for language modelling (autoregressive generation): when predicting the next token after position 5, you cannot let position 5 look at positions 6, 7, 8, etc. because those tokens do not exist yet at generation time. The mask sets all future scores to $-\infty$ before softmax, ensuring $\text{softmax}(-\infty) = 0$ — future positions receive zero attention weight. From a circuit analysis perspective, the causal mask means that all circuits we analyse are strictly feedforward in the position dimension: information flows from earlier positions to later ones, never backwards.

### 6.1 Annotated Transformer view

The core attention operation is:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

where $Q \in \mathbb{R}^{T_q \times d_k}$, $K \in \mathbb{R}^{T_k \times d_k}$, $V \in \mathbb{R}^{T_k \times d_v}$ are the query, key, and value matrices, and $T_q$, $T_k$ are the query and key sequence lengths (equal for self-attention).

The scaling factor $1/\sqrt{d_k}$ is critical: without it, in high dimensions the dot products $q \cdot k$ grow proportionally to $\sqrt{d_k}$, pushing softmax into a regime of near-zero gradients.

For **causal** (autoregressive) attention, an upper-triangular mask is applied before softmax:
$$\text{score}_{ij} = \begin{cases} q_i \cdot k_j / \sqrt{d_k} & j \leq i \\ -\infty & j > i \end{cases}$$

### 6.2 Circuit Framework annotation

**The attention pattern as an activation**: The output of softmax, $A = \text{softmax}(QK^T/\sqrt{d_k}) \in \mathbb{R}^{T \times T}$, is an **activation** — it depends on the input, not just the weights. It is *not* a parameter.

**The QK bilinear form**: Substituting $Q = X W_Q$ and $K = X W_K$ into the score formula:
$$\text{score}(i \to j) = \frac{(x_i W_Q)(x_j W_K)^T}{\sqrt{d_k}} = \frac{x_i W_Q W_K^T x_j^T}{\sqrt{d_k}} = \frac{x_i W_{QK} x_j^T}{\sqrt{d_k}}$$

where $W_{QK} := W_Q W_K^T \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$ is the **QK circuit matrix**. Once $W_{QK}$ is fixed, the attention score between any two residual stream vectors is determined by a single matrix product.

The key insight is that $q = x W_Q$ and $k = x W_K$ are **not** fundamental — they are intermediate computations in the calculation of $x W_{QK} x^T$. The circuit view bypasses them entirely.

**Low-rank structure**: $W_Q \in \mathbb{R}^{d_\text{model} \times d_\text{head}}$ and $W_K \in \mathbb{R}^{d_\text{model} \times d_\text{head}}$, so $W_{QK} = W_Q W_K^T$ has rank at most $d_\text{head} \ll d_\text{model}$.

**OV output**: After computing $A$, the head's output at position $i$ is:
$$\text{head output}_i = \sum_j A_{ij} x_j W_V W_O = \sum_j A_{ij} x_j W_{OV}$$

where $W_{OV} := W_V W_O \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$ is the **OV circuit matrix**: the combined read-then-write map.

---

## 7. Multi-Head Attention

### Plain Language Explanation

The motivation for running multiple attention heads in parallel, rather than a single large attention head, is analogous to the motivation for using many convolutional filters in a CNN rather than one large filter. Each head can specialise to a different type of relationship between tokens. One head might learn to connect verbs to their subjects. Another might learn to connect pronouns to the nouns they refer to. A third might focus on local syntax (words attending to adjacent words). By running many heads simultaneously and summing their outputs, the model can attend to many different aspects of the context at once.

The critical insight for interpretability is that the original concatenation + $W^O$ formulation disguises an additive structure. If you split the output projection matrix $W^O$ into one block per head, you see that MultiHead attention is identically equal to a sum of $H$ independent single-head attentions. Each head independently computes its own attention pattern and its own value-weighted output, and all $H$ outputs are summed and added to the residual stream. This additive structure means the heads **do not interact within a single layer** — they only interact through the shared residual stream that is passed to the next layer.

This additivity is essential for the circuit framework. It means you can always ask "what is head $(\ell, h)$'s contribution to the output?" and get an exact answer — even when 11 other heads are also running in the same layer. The heads are parallel workers, each adding their contribution to a shared sum. You never need to worry about one head "cancelling" another within the same layer in some nonlinear way; they just add.

### 7.1 Annotated Transformer view

Vaswani et al. propose running $H$ attention heads in parallel, each with its own $(W_Q^h, W_K^h, W_V^h)$ projections of smaller dimension $d_k = d_v = d_\text{model}/H$, and concatenating the outputs before a final linear projection $W^O$:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \, W^O$$

$$\text{head}_h = \text{Attention}(Q W_Q^h, \, K W_K^h, \, V W_V^h)$$

The original paper uses $H = 8$ heads, $d_\text{model} = 512$, giving $d_k = d_v = 64$. GPT-2 Small uses $H = 12$ heads, $d_\text{model} = 768$, giving $d_\text{head} = 64$.

### 7.2 Circuit Framework annotation

The concatenation + $W^O$ formulation obscures a crucial fact: **multi-head attention is equivalent to a sum of independent single-head attentions**.

To see this, split $W^O$ into $H$ blocks of rows, one per head:
$$W^O = \begin{bmatrix} W_{O,1} \\ \vdots \\ W_{O,H} \end{bmatrix} \Rightarrow \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W^O = \sum_{h=1}^H \text{head}_h \, W_{O,h}$$

So the total attention output added to the residual stream is:

$$\Delta_\text{attn} = \sum_{h=1}^{H} A^h \cdot (X W_V^h) \cdot W_{O,h} = \sum_{h=1}^{H} A^h \cdot X \underbrace{W_V^h W_{O,h}}_{W_{OV}^h}$$

Each head contributes an **independent additive term** to the residual stream. The heads do not interact within a layer — they interact only through the shared residual stream passed to the next layer.

**TransformerLens implementation**: Rather than storing $W^O$ as a single concatenated matrix, TransformerLens stores per-head matrices:

- `model.W_Q`: shape `[n_layers, n_heads, d_model, d_head]`
- `model.W_K`: shape `[n_layers, n_heads, d_model, d_head]`
- `model.W_V`: shape `[n_layers, n_heads, d_model, d_head]`
- `model.W_O`: shape `[n_layers, n_heads, d_head, d_model]`

This makes the per-head decomposition explicit.

---

## 8. The OV and QK Circuit Decomposition

### Plain Language Explanation

Every attention head performs two fundamentally different jobs simultaneously, and the circuit framework separates them cleanly. The first job is **routing**: deciding which positions should contribute to the current position's representation. This is determined by the QK circuit $W_{QK} = W_Q W_K^T$ — the bilinear form that computes attention scores. If you want to understand where a head attends (e.g., "does it always look at the previous word?"), you study $W_{QK}$.

The second job is **content transformation**: deciding what information to extract from the attended positions and write to the residual stream. This is determined by the OV circuit $W_{OV} = W_V W_O$ — the combined read-then-write map. If you want to understand what a head copies or transforms (e.g., "does attending to 'cat' boost the logit for 'cat', or for some related word?"), you study $W_{OV}$.

The crucial point is that these two jobs are largely independent. The QK circuit decides *where to look*, and the OV circuit decides *what to do with what you found*. This independence makes circuit analysis tractable: you can answer routing questions and content questions separately, with different mathematical tools. The QK circuit involves a bilinear form between query and key vectors; the OV circuit involves a linear map from source to destination. Both are analysable as matrix products, but they have different interpretive meanings.

### 8.1 Two independent circuits per head

Elhage et al. (2021) observe that each attention head consists of **two largely independent computations**:

| Circuit        | Formula                      | Role                                                            |
| -------------- | ---------------------------- | --------------------------------------------------------------- |
| **QK circuit** | $W_{QK}^h = W_Q^h (W_K^h)^T$ | Determines *where* information is moved (the attention pattern) |
| **OV circuit** | $W_{OV}^h = W_V^h W_O^h$     | Determines *what* information is moved (the content copied)     |

Both are $d_\text{model} \times d_\text{model}$ matrices, operating entirely in residual stream space. Both have rank at most $d_\text{head}$.

### 8.2 Paper convention vs. TransformerLens convention

The Elhage et al. paper uses column-vector convention (vectors are columns, matrices left-multiply). Their definitions are:
$$W_{QK}^{h,\text{paper}} = (W_Q^h)^T W_K^h, \qquad W_{OV}^{h,\text{paper}} = W_O^h W_V^h$$

TransformerLens uses row-vector convention (vectors are rows, matrices right-multiply). Converting:
$$W_{QK}^{h,\text{TL}} = W_Q^h (W_K^h)^T \qquad \text{(same matrix)}$$
$$W_{OV}^{h,\text{TL}} = W_V^h W_O^h \qquad \text{(transpose of paper's } W_{OV}\text{)}$$

The attention bilinear form gives the same scalar in both conventions:
$$\text{paper: } x_i^T W_{QK}^{\text{paper}} x_j = q_i^T k_j$$
$$\text{TL: } x_i W_{QK}^{\text{TL}} x_j^T = q_i \cdot k_j \quad \checkmark$$

### 8.3 Attention head output in circuit notation

For head $h$ in layer $\ell$, the contribution to the residual stream at position $i$ is:

$$\Delta x_i^{\ell,h} = \sum_{j \leq i} A^{\ell,h}_{ij} \cdot x_j^{(\ell-1)} W_{OV}^{\ell,h}$$

where $A^{\ell,h}_{ij} = \text{softmax}\!\left(\frac{x_i^{(\ell-1)} W_{QK}^{\ell,h} (x_j^{(\ell-1)})^T}{\sqrt{d_\text{head}}}\right)_j$.

---

## 9. Full Circuits: From Vocabulary to Vocabulary

### Plain Language Explanation

The $W_{QK}$ and $W_{OV}$ matrices we derived in the previous section operate on residual stream vectors, which have no fixed meaning. To understand what a head actually does to words, we need to connect these residual-stream operations to the vocabulary. We do this by sandwiching the circuit matrices between the embedding $W_E$ (which converts tokens into residual-stream vectors) and the unembedding $W_U$ (which converts residual-stream vectors back into logit predictions).

The result is called a "full circuit" because it completes the round-trip from vocabulary to vocabulary. The **full OV circuit** $W_E W_{OV}^h W_U$ is a $d_\text{vocab} \times d_\text{vocab}$ matrix where entry $(s, d)$ answers: "if a position fully attends to source token $s$, by how much does the logit for destination token $d$ change?" This is the most interpretable form of the OV circuit: it is stated entirely in terms of tokens, with no residual-stream coordinates to worry about.

Similarly, the **full QK circuit** $W_E W_{QK}^h W_E^T$ is a $d_\text{vocab} \times d_\text{vocab}$ matrix where entry $(t_i, t_j)$ is the unnormalized content-based attention score between a position holding token $t_i$ and a position holding token $t_j$. Large values mean "query token $t_i$ tends to attend to key token $t_j$." This is the interpretable form of the QK circuit. Both full circuits are large but low-rank (rank at most $d_\text{head} = 64$), so they are efficiently represented as products of thin matrices using the FactoredMatrix class.

### 9.1 The privileged basis problem revisited

$W_{QK}^h$ and $W_{OV}^h$ act on residual stream vectors, which (as argued in Section 3) have no privileged basis. To make the circuits interpretable, we must compose them with $W_E$ and $W_U$ to connect to the vocabulary.

### 9.2 Full OV circuit

$$\text{Full OV}^h := W_E \, W_{OV}^h \, W_U \in \mathbb{R}^{d_\text{vocab} \times d_\text{vocab}}$$

Row $s$ of this matrix is the change in output logits at the destination position when it attends fully to source token $s$. Entry $(s, d)$ answers:

> "If the destination fully attends to source token $s$, by how much does the logit for token $d$ change?"

Large values at $(s, s)$ (diagonal) indicate a **copying head**: attending to token $s$ promotes predicting $s$. Large off-diagonal values at $(s, d)$ indicate that source token $s$ promotes destination prediction $d$ — potentially translation, syntactic pattern completion, or other semantic relationships.

This matrix is $d_\text{vocab} \times d_\text{vocab}$ but has rank at most $d_\text{head}$ (the bottleneck). Materializing it directly would require $\sim 2.5$ billion floats for GPT-2 vocabulary — hence the `FactoredMatrix` class.

### 9.3 Full QK circuit

$$\text{Full QK}^h := W_E \, W_{QK}^h \, W_E^T \in \mathbb{R}^{d_\text{vocab} \times d_\text{vocab}}$$

Entry $(t_i, t_j)$ is the unnormalized attention score from a query at a position whose token is $t_i$, to a key at a position whose token is $t_j$. This captures purely **content-based attention**: which tokens attend to which other tokens regardless of position.

### 9.4 Positional QK circuit

$$W_\text{pos} \, W_{QK}^h \, W_\text{pos}^T \in \mathbb{R}^{n_\text{ctx} \times n_\text{ctx}}$$

Entry $(i, j)$ is the contribution to the attention score from position $i$ to position $j$, based purely on position. For a **previous-token head**, entry $(i, i-1)$ should dominate: position $i$ attends to position $i-1$ regardless of what tokens are there.

---

## 10. Position-wise Feed-Forward Networks

### Plain Language Explanation

The position-wise feed-forward network (FFN) is the second major component of each transformer layer, after attention. Unlike attention, which operates across all positions (computing pairwise interactions), the FFN is applied independently to each position: the same function is applied to each token's residual stream vector without any cross-position communication. This is why it is called "position-wise": each position gets its own independent FFN pass.

Architecturally, the FFN is a two-layer MLP: a first linear layer expands the dimension from $d_\text{model}$ to $4 \times d_\text{model}$ (the "inner dimension"), a non-linear activation function (ReLU in the original paper, GELU in modern variants) introduces nonlinearity, and a second linear layer projects back down to $d_\text{model}$. The expansion by a factor of 4 is somewhat arbitrary — it was found to work well empirically. For $d_\text{model} = 512$, the FFN has about $2 \times 512 \times 2048 = 2$ million parameters per layer, accounting for roughly two thirds of the model's parameters.

From a circuit analysis perspective, the FFN is the most challenging component. The ReLU or GELU activation function introduces a genuine nonlinearity: whether a particular neuron fires depends on the value of the input to it, which in turn depends on the entire residual stream history. Unlike attention, where the final output can be expressed as a product of fixed weight matrices regardless of the input, the FFN's output involves a nonlinear gating function that cannot be captured by a single matrix product. Elhage et al. (2021) explicitly describe this as a "major weakness" of their framework. The superposition hypothesis (Elhage et al., 2022b) proposes that FFN neurons may be encoding many features simultaneously in overlapping directions, further complicating interpretation.

### 10.1 Annotated Transformer view

Each transformer layer also contains a **position-wise feed-forward network** (FFN), applied independently to each position:

$$\text{FFN}(x) = \max(0, \, x W_1 + b_1) \, W_2 + b_2$$

where $W_1 \in \mathbb{R}^{d_\text{model} \times d_\text{ff}}$, $W_2 \in \mathbb{R}^{d_\text{ff} \times d_\text{model}}$, and $d_\text{ff} = 4 d_\text{model}$ (Vaswani et al. use $d_\text{model} = 512$, $d_\text{ff} = 2048$). The nonlinearity is ReLU in the original paper; modern models use GELU.

The FFN is where most of the model's parameters live: $W_1$ and $W_2$ together account for $2 \times d_\text{model} \times d_\text{ff} = 8 d_\text{model}^2$ parameters, compared to $4 d_\text{model}^2 / H$ per attention head.

### 10.2 Circuit Framework annotation

The FFN is the **main difficulty** for circuit analysis. Its output at each position is:
$$\Delta x_s^\text{FFN} = \sigma(x_s W_1 + b_1) W_2$$

The nonlinearity $\sigma$ (ReLU or GELU) breaks the linear structure. Unlike attention heads, the FFN's contribution cannot be reduced to a product of interpretable matrices that connect vocabulary tokens to logits.

Elhage et al. (2021) acknowledge this directly:

> "MLPs are a major weakness of our work... we are largely unable to interpret what they are doing."

The **superposition hypothesis** (Elhage et al., 2022, a follow-up paper) proposes that MLP neurons represent many features simultaneously in superposition, exploiting the high-dimensional space to pack more information than the number of neurons.

For the purpose of circuit analysis, it is common to:

1. Study attention-only models (no FFN layers) first.
2. Treat the FFN as a black box and focus on the attention circuit.
3. Use neuron-level analysis or sparse probing to investigate FFNs separately.

---

## 11. Layer Normalization

### Plain Language Explanation

Layer normalisation is a technique for stabilising the training of deep networks. Without it, the activations in deep networks tend to drift: the mean and variance of activations at layer 10 might be very different from those at layer 1, causing gradients to explode or vanish. Layer normalisation fixes this by normalising each activation vector to have mean 0 and standard deviation 1 (measured across the $d_\text{model}$ dimensions), then applying learned per-dimension scale $\gamma$ and shift $\beta$ parameters.

Computationally, LayerNorm is a simple operation: subtract the mean, divide by the standard deviation, multiply by $\gamma$, add $\beta$. The learned $\gamma$ and $\beta$ allow the network to undo the normalisation if needed (e.g., if a dimension's scale is important for downstream layers). In the original Vaswani et al. paper, LayerNorm is applied after each residual addition ("post-norm"). In most modern models (GPT-2, LLaMA), it is applied before each sub-layer input ("pre-norm"), which tends to be more stable to train.

From a circuit analysis perspective, LayerNorm is a mild complication. The division by $\sigma$ (the standard deviation of the activation vector) is nonlinear: $\sigma$ depends on $x$, so the normalised value is a nonlinear function of $x$. However, for inputs with approximately constant norm (which is often the case in practice), LayerNorm is approximately linear — it just rescales the input by a constant factor. The Mathematical Framework handles this by either folding the LayerNorm into adjacent weight matrices (as an approximation) or by studying models without LayerNorm (the toy models in ARENA exercises have no LayerNorm). The circuit conclusions are qualitatively robust to this approximation.

### 11.1 Annotated Transformer view

Layer normalisation (Ba et al., 2016) is applied before each sublayer in the "pre-norm" variant used by most modern models (GPT-2, Llama, etc.) or after in the "post-norm" of the original Vaswani et al. paper:

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \odot \gamma + \beta$$

where $\mu$ and $\sigma$ are the mean and standard deviation computed over the $d_\text{model}$ dimension of $x$, and $\gamma, \beta \in \mathbb{R}^{d_\text{model}}$ are learned affine parameters.

### 11.2 Circuit Framework annotation

Layer normalisation introduces a **mild nonlinearity** (the normalization by $\sigma$ depends on $x$) that technically breaks the linear structure. However:

- In the limit where inputs have approximately constant norm, LayerNorm is approximately linear.
- For analytical purposes, the Mathematical Framework often **folds LayerNorm into adjacent weight matrices** or ignores it as a second-order effect.
- The toy models used in ARENA exercises (the two-layer attention-only model) have **no LayerNorm**, making the algebra exact.

In practice, interpretability researchers (including Nanda's TransformerLens) include LayerNorm but treat the circuit analysis as an approximation. The conclusions about W_QK, W_OV, and composition are qualitatively robust to LayerNorm's presence.

---

## 12. Encoder–Decoder Attention and Cross-Attention

### Plain Language Explanation

In the encoder–decoder architecture of Vaswani et al., the decoder must look at two things simultaneously: the English words it has already generated (via self-attention), and the French sentence it is trying to translate (via cross-attention). Cross-attention is mechanically identical to self-attention, except that the queries come from the decoder's residual stream while the keys and values come from the encoder's final hidden states. The decoder "asks" (via its query) and the encoder "answers" (via its keys and values).

From an interpretability standpoint, cross-attention is particularly interesting because it is the only place where information flows from the source language to the target language. The full OV circuit for cross-attention, $W_E^\text{src} W_{OV}^\text{cross} W_U^\text{tgt}$, is literally a learned translation table: entry $(s, d)$ measures how much attending to source word $s$ promotes predicting target word $d$. This is a direct, algebraic representation of what the model learned about the translation task.

The full QK circuit for cross-attention, $W_E^\text{dec} W_{QK}^\text{cross} (W_E^\text{enc})^T$, tells you which target tokens attend to which source tokens based on their content. A translation model might learn that questions words ("what", "when", "who") in the target language attend to the corresponding question words in the source language, or that verb phrases attend to their source-language equivalents. Plotting this matrix for a trained translation model gives a direct view into the model's learned alignment.

### 12.1 Annotated Transformer view

The decoder's middle sublayer performs **cross-attention**: the queries come from the decoder's residual stream, while the keys and values come from the encoder's output. This is identical in form to self-attention, except that $Q$ and $(K, V)$ originate from different sequences:

$$\text{CrossAttn}(x_\text{dec}, x_\text{enc}) = \text{softmax}\!\left(\frac{(x_\text{dec} W_Q)(x_\text{enc} W_K)^T}{\sqrt{d_k}}\right) \cdot x_\text{enc} W_V W_O$$

### 12.2 Circuit Framework annotation

In the circuit framework view, cross-attention is analyzed with the same tools:
$$W_{QK}^\text{cross} = W_Q W_K^T \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$$
$$W_{OV}^\text{cross} = W_V W_O \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$$

The QK circuit now computes $x_\text{dec} W_{QK}^\text{cross} x_\text{enc}^T$: it determines which encoder tokens each decoder position attends to. The OV circuit determines what information is extracted from the attended encoder positions.

For encoder–decoder models, the **full OV circuit** maps source vocabulary tokens to target logit changes:
$$W_E^\text{src} \, W_{OV}^\text{cross} \, W_U^\text{tgt} \in \mathbb{R}^{d_\text{vocab}^\text{src} \times d_\text{vocab}^\text{tgt}}$$

This is the matrix describing how strongly each source token promotes each target token prediction — a direct representation of the learned translation table.

---

## 13. The Path Expansion

### Plain Language Explanation

The path expansion is the Mathematical Framework's most powerful tool for understanding multi-layer models. The core idea is simple: because every component adds to the residual stream, and the final logit is a linear function of the final residual stream, the final logit is a sum of contributions from every component. Each contribution is an "additive path" through the network: a sequence of choices about which residual stream terms to follow.

In a one-layer model, there are only two types of paths: the direct path (embedding → unembedding, no attention) and the single-layer head paths (embedding → attention head → unembedding). In a two-layer model, paths become more complex: a layer-2 head's computation can depend on what a layer-1 head wrote to the residual stream, creating cross-layer terms. These cross-layer paths are the "composition terms" that give two-layer models their extra expressive power.

The path expansion is exact — it is not an approximation or a linear regression. It is a straightforward consequence of linearity: the sum of components writes to the residual stream, and the logit is a linear function of the residual stream, so the logit is the sum of a linear function applied to each component's contribution. This exactness is what makes the decomposition so useful: you can point to a specific path and say "this path contributed exactly +2.3 logits to the correct token on this input," with no error bars.

### 13.1 Logits as a sum over paths

For an $L$-layer attention-only model with $H$ heads per layer, let $o^{\ell,h}$ denote the output written to the residual stream by head $(\ell, h)$. Because all writes are additive:

$$x_s^{(L)} = x_s^{(0)} + \sum_{\ell=1}^{L} \sum_{h=1}^{H} o_s^{\ell,h}$$

$$\text{logits}_s = x_s^{(L)} W_U = x_s^{(0)} W_U + \sum_{\ell,h} o_s^{\ell,h} W_U$$

Each term $o_s^{\ell,h} W_U$ is the **direct logit contribution** of head $(\ell,h)$ at position $s$. This decomposition is exact.

### 13.2 The path expansion for two-layer models

For two layers, the residual stream fed to layer 2 includes the layer-1 outputs:
$$x^{(1)} = x^{(0)} + \sum_{h \in L_1} o^{1,h}$$

So the attention pattern of a layer-2 head $h_2$ reads keys and queries from $x^{(1)}$, expanding into:

$$\text{score}^{2,h_2}(i \to j) = \underbrace{x_i^{(0)} W_{QK}^{2,h_2} (x_j^{(0)})^T}_{\text{no composition}} + \underbrace{\sum_{h_1} o_i^{1,h_1} W_{QK}^{2,h_2} (x_j^{(0)})^T}_{\text{Q-composition}} + \underbrace{\sum_{h_1} x_i^{(0)} W_{QK}^{2,h_2} (o_j^{1,h_1})^T}_{\text{K-composition}} + \ldots$$

The full logit decomposition for a two-layer model consists of:

**(1) Direct path**: $x^{(0)} W_U$ — embedding to logit with no attention.

**(2) Layer-1 head contributions**: $\sum_{h_1} o^{1,h_1} W_U$ — each layer-1 head's direct write to the logits.

**(3) Layer-2 head contributions (no composition)**: layer-2 heads attending based purely on $x^{(0)}$.

**(4) Composition terms**: layer-2 heads attending or reading values through layer-1 outputs.

---

## 14. Zero-Layer Transformers

### Plain Language Explanation

A zero-layer transformer is the simplest possible transformer: just an embedding layer and an unembedding layer, with no attention and no feed-forward network in between. It can only make predictions based on the single current token — it has no mechanism to look at other tokens in the sequence.

The computation is just: look up the current token in $W_E$ to get a 768-dimensional vector, then project through $W_U$ to get logits. This is exactly a **bigram language model**: it predicts the probability of the next word given only the current word. The matrix product $W_E W_U$ encodes the bigram statistics of the training corpus — entry $(s, d)$ is proportional to $\log P(\text{next token} = d \mid \text{current token} = s)$, after the model is trained.

This zero-layer analysis is not just a pedagogical toy. In multi-layer models, the zero-layer term (the "direct path" $x_0 W_U$) is always present as one of the additive contributions. It represents the bigram prior that the model falls back on when other components are not active. Understanding the direct path first makes it easier to understand what the attention heads add on top of it — they can be thought of as corrections to the bigram prior, making the prediction more context-sensitive.

### 14.1 Definition

A zero-layer transformer has no attention heads and no FFN — just embeddings and an unembedding:
$$\text{logits} = x_s^{(0)} W_U = (t_s W_E) W_U$$

### 14.2 Circuit Framework analysis

The **direct path** $W_E W_U \in \mathbb{R}^{d_\text{vocab} \times d_\text{vocab}}$ is the entire model. Entry $(s, d)$ gives the logit assigned to predicting token $d$ when the input token is $s$. This is learned to approximate the unigram distribution: how often is token $d$ the next token when token $s$ was just seen?

More precisely, this is a **bigram model**: it models $P(\text{next} \mid \text{current})$ for each token pair. It cannot condition on any context further back than one token.

**Key equation**: For a zero-layer model, all the interpretable information lives in $W_E W_U$. This matrix is rank at most $d_\text{model}$, compressing the full $d_\text{vocab} \times d_\text{vocab}$ co-occurrence matrix into a low-rank factorization.

---

## 15. One-Layer Transformers

### Plain Language Explanation

A one-layer transformer adds a single layer of attention on top of the bigram direct path. The resulting model can implement **skip-trigrams**: predictions that depend on two specific tokens in the context (the current token and one earlier token), but not on the positions of those tokens or on what happened between them. For example, "I am going to [attend to 'Paris'] [predict 'France']" is a skip-trigram: the model learns that when the current context contains "Paris" and the current token is "going to", the next token might be related to France.

The exact formula for the one-layer model's predictions decomposes into: the bigram prior (direct path $t_s W_E W_U$) plus, for each attention head, a weighted sum of what each source position $s'$ contributes via the full OV circuit. The weight is the attention probability $A_{s,s'}^h$ — how much the head attends to position $s'$. The value is row $s'$ of the full OV matrix — what logit changes would result from attending to the token at $s'$.

One-layer models cannot implement induction (the "repeat the pattern" behavior). To understand why: induction requires the model to find positions in the context that previously followed the current token. This requires comparing two positions that are neither adjacent nor the current position — a global search over context that requires knowing the content of positions relative to each other. In a one-layer model, the QK circuit can only compare the current query token with key tokens, based on their raw embeddings. It cannot access any information about what surrounds those key positions. That requires two layers.

### 15.1 Complete logit formula

For a one-layer attention-only model with $H$ heads, the logits at position $s$ are:

$$\text{logits}_s = \underbrace{x_s^{(0)} W_U}_{\text{direct path}} + \sum_{h=1}^{H} \sum_{s' \leq s} A^h_{s,s'} \underbrace{x_{s'}^{(0)} W_{OV}^h W_U}_{\text{row } s'\text{ of full OV}^h}$$

This is exact. The attention pattern $A^h$ is the softmax of $x^{(0)} W_{QK}^h (x^{(0)})^T / \sqrt{d_\text{head}}$.

### 15.2 Skip-trigrams

The one-layer model can implement **skip-trigram** predictions: "if the recent context contains token $A$ at some position, and the current token is $B$, predict token $C$."

The mechanism: the QK circuit $W_E W_{QK}^h W_E^T$ determines that queries formed from $B$-embeddings attend strongly to positions holding $A$. The OV circuit $W_E W_{OV}^h W_U$ determines that attending to $A$ boosts the logit for $C$.

Combined: the model implements $P(C \mid \ldots A \ldots B)$ via the bilinear interaction. The "skip" (arbitrary gap between $A$ and $B$) is possible because attention can attend to any earlier position, not just the immediately preceding one.

**Expressive limits**: a one-layer model cannot implement induction (predicting $B$ given that $A \to B$ appeared earlier in context) because that would require the attention pattern to depend on the *content* of positions relative to each other — which requires two layers.

### 15.3 The direct path as a bigram table

The direct path $W_E W_U$ is additive with the head contributions. In a trained model, the direct path typically implements a **bigram prior** (predict common continuations of the current token), while the attention heads implement context-sensitive corrections.

---

## 16. Two-Layer Transformers

### Plain Language Explanation

Moving from one layer to two layers introduces composition: the possibility that one attention head's output is used as input to another head's computation. This is qualitatively different from having two independent one-layer models stacked together, because the second layer's heads are not just seeing the original embeddings — they are seeing those embeddings modified by the first layer's heads.

There are three ways composition can happen. In **K-composition**, the first head's OV output influences the second head's key computation — effectively changing "what position $j$ advertises" based on what the first head computed. In **Q-composition**, the first head's OV output influences the second head's query computation — changing "what the destination is searching for." In **V-composition**, the first head's OV output influences the second head's value computation — changing what information is extracted once the destination decides where to attend.

Each type of composition creates a new type of circuit that is impossible in one layer. K-composition enables induction heads (as described in detail in Section 17): the first head writes "my left neighbour is X" into each position's residual stream, and the second head uses that information in its key, allowing it to find positions where X was the left neighbour. Q-composition can enable more complex behaviours where the question being asked by the second head depends on what the first head found. V-composition creates a two-step information chain where information is transformed twice before being written to the output logits.

### 16.1 Composition types

Two-layer models introduce **composition**: a layer-2 head's computation can depend on a layer-1 head's output. There are three types:

**K-composition**: Layer-1 head $h_1$ writes to the residual stream; layer-2 head $h_2$ reads this via its key projection. The effective key at position $j$ for head $h_2$ is:
$$k_j^{2,h_2} \approx x_j^{(0)} W_K^{h_2} + o_j^{1,h_1} W_K^{h_2}$$
The second term gives $h_2$'s QK circuit access to $h_1$'s OV output. The **K-composition circuit** is:
$$W_{OV}^{h_1} W_{QK}^{h_2} \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$$
and the full vocabulary-level version is:
$$W_E W_{OV}^{h_1} W_{QK}^{h_2} W_E^T \in \mathbb{R}^{d_\text{vocab} \times d_\text{vocab}}$$

**Q-composition**: Layer-1 head $h_1$ influences $h_2$'s query projection. The Q-composition circuit is:
$$W_{OV}^{h_1} W_{QK}^{h_2}$$
(same matrix as K-composition, but now $h_1$'s output feeds the query side rather than the key side).

**V-composition**: Layer-1 head $h_1$ influences what $h_2$ reads as its values. The V-composition circuit is:
$$W_{OV}^{h_1} W_{OV}^{h_2}$$
This is the composition of two OV circuits: $h_1$ reads from the embedding and transforms it; $h_2$ reads $h_1$'s output and transforms it again.

### 16.2 Virtual weights

When two heads compose, their composed weight matrix is called a **virtual weight**:
$$W_\text{virtual} = W_{OV}^{h_1} W_{QK}^{h_2} \quad \text{(K or Q composition)}$$
$$W_\text{virtual} = W_{OV}^{h_1} W_{OV}^{h_2} \quad \text{(V composition)}$$

Virtual weights are interpretable in the same way as single-head circuits. The composition of two linear functions is another linear function — two composing heads act like a single more powerful head.

---

## 17. Induction Heads

### Plain Language Explanation

Induction heads are the simplest example of a non-trivial algorithm that emerges from the composition of two attention heads in two layers. The algorithm they implement is in-context copying: if the sequence has seen the pattern "A B" at some earlier point, and "A" appears again, the induction head predicts "B" will follow. This is called "induction" because it is the simplest form of learning from the context: inducing a rule from one example ("A is followed by B") and applying it immediately to predict the next occurrence.

What makes induction heads striking is that this pattern-matching happens purely from the context window, not from memorised training data. You can show induction heads completely novel tokens — emoji, rare Unicode characters, random byte sequences — that they have never seen during training, and they still correctly predict the repeat. The model is not recalling a memorised fact; it is running a circuit that reads the current context and applies it. This is qualitatively different from a traditional lookup table or language model, and it is one of the first documented examples of a mechanistic in-context learning algorithm.

To understand how this is implemented, think through the two-step process carefully. After layer 0's previous-token head runs, every position $j$'s residual stream contains not just the content of $t_j$ but also an encoding of $t_{j-1}$. Now in layer 1, head 1.4 computes keys from this modified residual stream. The key at position $j$ effectively encodes "what was to my left?" — i.e., $t_{j-1}$. The query at the second occurrence of $A$ asks (via the K-composition circuit) "find the position where what was to the left is $A$." The position that satisfies this is position $i+1$ (right after the first occurrence of $A$), which has $t_i = A$ to its left. Head 1.4 attends to position $i+1$, copies token $t_{i+1} = B$ via its OV circuit, and predicts $B$.

### 17.1 What they do

An **induction head** (Elhage et al., 2021; Olsson et al., 2022) is an attention head that implements in-context sequence copying: given that $A \to B$ appeared earlier in the context, when $A$ appears again it attends back to the first $B$ and predicts $B$.

This is qualitatively more expressive than a one-layer skip-trigram because the head does not memorize $A \to B$ from training data — it reads the association dynamically from the current context. The model can generalise to completely novel $A \to B$ pairs that never appeared in training.

### 17.2 Circuit implementation

The induction circuit requires exactly two attention heads in two layers:

**Previous-token head** (layer 0, e.g. head 0.7): attends from position $i$ to position $i-1$. Its QK circuit is dominated by $W_\text{pos} W_{QK}^{0.7} W_\text{pos}^T$, which has large values just below the diagonal. Its OV circuit copies the previous token's representation to the current residual stream position.

**Induction head** (layer 1, e.g. head 1.4): uses **K-composition** with the previous-token head. The full K-composition circuit for the induction behavior is:

$$W_E W_{OV}^{0.7} W_{QK}^{1.4} W_E^T \in \mathbb{R}^{d_\text{vocab} \times d_\text{vocab}}$$

Entry $(B, A)$ of this matrix is large: when the second $A$ queries, it finds the strongest key match at the position of the first $B$, causing the induction head to attend to $B$ and copy it forward via its own OV circuit:

$$W_E W_{OV}^{1.4} W_U \qquad \text{(near-diagonal copying matrix)}$$

### 17.3 Why two layers are necessary

A single-layer attention head computes $\text{score}(i \to j) = f(x_i, x_j)$ — a function of the current token embeddings at positions $i$ and $j$ only. Induction requires scoring based on "what token follows $j$?" — a property of position $j+1$, not $j$. This is impossible in one layer.

The K-composition mechanism is what makes this possible: head 0.7 writes information about position $j-1$'s token into position $j$'s residual stream; head 1.4 then reads this as part of position $j$'s key. From head 1.4's perspective, position $j$'s key encodes "the token before me" — enabling the offset lookup.

---

## 18. Composition Scores

### Plain Language Explanation

The composition score is a diagnostic that lets you measure, purely from the model's weights and without running any inputs, whether two specific heads are wired together. The intuition comes from thinking about subspaces: each attention head writes its output into a $d_\text{head}$-dimensional subspace of the $d_\text{model}$-dimensional residual stream, and reads from a different $d_\text{head}$-dimensional subspace. For the first head's output to influence the second head's computation, these subspaces must overlap.

In a high-dimensional space like $\mathbb{R}^{768}$, two random 64-dimensional subspaces are almost certainly orthogonal — their overlap is essentially zero. This is a consequence of the concentration of measure phenomenon in high dimensions. So if we find two heads whose subspaces do overlap, it is almost certainly because training made them overlap on purpose. The composition score quantifies this overlap as a normalised Frobenius norm of the product $W_A W_B$: if the column space of $W_A$ and the row space of $W_B$ are perfectly aligned, the product has maximal norm; if they are orthogonal, the product is zero.

The three types of composition score (Q, K, V) correspond to the three ways a second head can depend on a first head's output: via its query computation (Q-comp), key computation (K-comp), or value computation (V-comp). The K-composition between head 0.7 and head 1.4 in the two-layer induction model is predicted by the circuit analysis to be high, and this is indeed what you find when you compute the scores. Heads that are not part of any circuit should have composition scores near the random baseline of $\approx 1/\sqrt{768} \approx 0.036$.

### 18.1 Definition

To measure whether two heads are genuinely composing, Elhage et al. (2021) define the **composition score**:

$$C(W_A, W_B) = \frac{\|W_A W_B\|_F}{\|W_A\|_F \cdot \|W_B\|_F}$$

where $\|\cdot\|_F$ is the Frobenius norm $\|M\|_F = \sqrt{\sum_{i,j} M_{ij}^2} = \sqrt{\sum_k \sigma_k^2}$ (square root of the sum of squared singular values).

### 18.2 Geometric interpretation

Each attention head operates in a $d_\text{head}$-dimensional subspace of the $d_\text{model}$-dimensional residual stream. For $d_\text{head} = 64$ and $d_\text{model} = 768$, any two random heads occupy nearly orthogonal subspaces (in expectation their composition score is $\approx 1/\sqrt{d_\text{model}} \approx 0.036$).

The composition score is large when $W_A$'s column space (what it writes into the residual stream) aligns with $W_B$'s row space (what it reads from the residual stream). When this alignment is deliberate — the heads have learned to share a subspace — the score is significantly above the $1/\sqrt{d_\text{model}}$ baseline.

### 18.3 The three composition types

| Composition       | $W_A$ (earlier head)                 | $W_B$ (later head)                          |
| ----------------- | ------------------------------------ | ------------------------------------------- |
| **Q-composition** | $W_{OV}^{h_A} = W_V^{h_A} W_O^{h_A}$ | $W_{QK}^{h_B} = W_Q^{h_B}(W_K^{h_B})^T$     |
| **K-composition** | $W_{OV}^{h_A} = W_V^{h_A} W_O^{h_A}$ | $(W_{QK}^{h_B})^T = W_K^{h_B}(W_Q^{h_B})^T$ |
| **V-composition** | $W_{OV}^{h_A} = W_V^{h_A} W_O^{h_A}$ | $W_{OV}^{h_B} = W_V^{h_B} W_O^{h_B}$        |

### 18.4 Why the Frobenius norm?

The Frobenius norm satisfies $\|M\|_F^2 = \sum_k \sigma_k(M)^2$. The composition score $\|AB\|_F / (\|A\|_F \|B\|_F)$ measures the **singular value alignment** between $A$ and $B$. If the top singular vectors of $A$ align with those of $B$, the product is large. If they are orthogonal, the product is small.

This metric is computationally tractable: $\|W_A W_B\|_F$ can be computed without materializing $W_A W_B$ directly, using the FactoredMatrix class.

---

## 19. Training

### Plain Language Explanation

The training details of the original Transformer — the Adam optimiser with warmup, label smoothing — are well-established engineering choices that Vaswani et al. validated empirically. They are not specific to the circuit framework, but they have consequences for what circuits the model learns.

The Adam optimiser adapts the learning rate per parameter, which helps different parameters converge at different rates. The warmup schedule (linearly increasing learning rate for the first 4000 steps, then decaying as $1/\sqrt{\text{step}}$) helps avoid instability at the start of training, when the model's parameters are far from their final values and gradients can be large. Label smoothing prevents overconfidence: instead of training the model to predict "100% probability on the correct token, 0% on everything else," it trains it to predict "90% on the correct token, 0.01% on everything else." This improves calibration and generalisation.

From a circuits perspective, the most important training phenomenon is the **phase transition in induction head formation**. Olsson et al. (2022) showed that induction heads in two-layer (and larger) transformers do not form gradually — they appear suddenly, at a specific point during training, accompanied by a visible kink in the loss curve. Before the phase transition, the model has no induction heads and no in-context learning ability. After the phase transition, it has both. This suggests that the induction circuit is a discrete structure that either exists or doesn't, rather than a continuous capability that improves gradually. Understanding these phase transitions is an active area of mechanistic interpretability research.

### 19.1 Annotated Transformer view

Vaswani et al. use the **Adam optimizer** (Kingma & Ba, 2015) with $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$, and a learning rate schedule with **warmup** followed by inverse square-root decay:

$$\text{lr}(\text{step}) = d_\text{model}^{-0.5} \cdot \min\!\left(\text{step}^{-0.5},\; \text{step} \cdot \text{warmup}^{-1.5}\right)$$

with $\text{warmup} = 4000$ steps.

**Label smoothing** ($\varepsilon_{ls} = 0.1$) is applied: rather than a one-hot target distribution, the model is trained to assign probability $1 - \varepsilon_{ls}$ to the correct token and $\varepsilon_{ls} / (d_\text{vocab} - 1)$ to each other token. This prevents the model from becoming overconfident and improves calibration.

### 19.2 Circuit Framework annotation

Training dynamics are largely outside the scope of the Mathematical Framework, which focuses on *post-training* weight analysis. However, some connections are worth noting:

**Phase transitions**: Elhage et al. (2022, "In-Context Learning and Induction Heads") observe that **induction heads form suddenly** during training at a distinct phase transition (around 2–4 billion training tokens for GPT-2-scale models). The loss curve shows a visible bump when they appear. This phase transition is correlated with the emergence of in-context learning ability.

**Superposition and training**: As models train, they may learn to represent multiple features in superposition within the residual stream — packing more information into fewer dimensions than naive dimensionality counting would allow.

**Logit attribution as a training signal proxy**: The logit attribution decomposition (Section 13) can be computed at any checkpoint during training, providing a window into how different heads' roles evolve.

---

## 20. Summary: Notation Rosetta Stone

The table below maps every key quantity between the three notations: Vaswani et al. (2017), the Annotated Transformer implementation, and the Mathematical Framework (Elhage et al., 2021). Use this table when reading any of the three sources to quickly translate between them.

| Concept               | Vaswani et al. / Annotated Transformer             | Mathematical Framework (Elhage et al.)                                  |
| --------------------- | -------------------------------------------------- | ----------------------------------------------------------------------- |
| Token embedding       | $W_E$, `Embeddings`                                | $W_E \in \mathbb{R}^{d_\text{vocab} \times d_\text{model}}$             |
| Token unembedding     | Pre-softmax linear                                 | $W_U \in \mathbb{R}^{d_\text{model} \times d_\text{vocab}}$             |
| Positional encoding   | $\text{PE}(\text{pos}, i)$, `PositionalEncoding`   | $W_\text{pos} \in \mathbb{R}^{n_\text{ctx} \times d_\text{model}}$      |
| Query projection      | $W_i^Q \in \mathbb{R}^{d_\text{model} \times d_k}$ | $W_Q^h \in \mathbb{R}^{d_\text{model} \times d_\text{head}}$            |
| Key projection        | $W_i^K \in \mathbb{R}^{d_\text{model} \times d_k}$ | $W_K^h \in \mathbb{R}^{d_\text{model} \times d_\text{head}}$            |
| Value projection      | $W_i^V \in \mathbb{R}^{d_\text{model} \times d_v}$ | $W_V^h \in \mathbb{R}^{d_\text{model} \times d_\text{head}}$            |
| Output projection     | $W^O \in \mathbb{R}^{H d_v \times d_\text{model}}$ | $W_O^h \in \mathbb{R}^{d_\text{head} \times d_\text{model}}$ (per head) |
| Attention scores      | $QK^T / \sqrt{d_k}$                                | $x W_{QK}^h x^T / \sqrt{d_\text{head}}$                                 |
| Attention weights     | $\text{softmax}(QK^T / \sqrt{d_k})$                | $A^h \in \mathbb{R}^{T \times T}$                                       |
| Head output           | $\text{Attention}(Q,K,V) = A^h V$                  | $A^h X W_V^h$                                                           |
| After output proj.    | $\text{head}_h W_{O,h}$                            | $A^h X W_{OV}^h$                                                        |
| QK circuit            | $W_i^Q (W_i^K)^T$                                  | $W_{QK}^h = W_Q^h (W_K^h)^T$                                            |
| OV circuit            | $W_i^V W_{O,h}$ (implicit)                         | $W_{OV}^h = W_V^h W_O^h$                                                |
| Full OV circuit       | —                                                  | $W_E W_{OV}^h W_U$                                                      |
| Full QK circuit       | —                                                  | $W_E W_{QK}^h W_E^T$                                                    |
| Residual stream       | Residual connection: $x + f(x)$                    | $x^{(\ell)} = x^{(0)} + \sum_{\ell',h} o^{\ell',h}$                     |
| Logit decomposition   | —                                                  | $\text{logits} = x^{(0)} W_U + \sum_{\ell,h} o^{\ell,h} W_U$            |
| K-composition circuit | —                                                  | $W_E W_{OV}^{h_1} W_{QK}^{h_2} W_E^T$                                   |
| Composition score     | —                                                  | $\|W_A W_B\|_F / (\|W_A\|_F \|W_B\|_F)$                                 |
| Virtual weight        | —                                                  | $W_{OV}^{h_1} W_{QK}^{h_2}$                                             |
| Privileged basis      | Input/output embeddings                            | $W_E$ (input) and $W_U$ (output)                                        |
| Copying score         | —                                                  | diagonal of $W_E W_{OV}^h W_U$                                          |

---

## References

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need*. NeurIPS 2017.

- Rush, A. (2018/2022). *The Annotated Transformer*. Harvard NLP. https://nlp.seas.harvard.edu/annotated-transformer/

- Elhage, N., Nanda, N., Olah, C., et al. (2021). *A Mathematical Framework for Transformer Circuits*. Transformer Circuits Thread. https://transformer-circuits.pub/2021/framework/index.html

- Olsson, C., Elhage, N., Nanda, N., et al. (2022). *In-Context Learning and Induction Heads*. Transformer Circuits Thread. https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html

- Elhage, N., et al. (2022). *Toy Models of Superposition*. Transformer Circuits Thread. https://transformer-circuits.pub/2022/toy_model/index.html

- Ba, J., Kiros, J., & Hinton, G. (2016). *Layer Normalization*. arXiv:1607.06450.

- Kingma, D. & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR 2015.
