# CTMRG — 2D Ising against Onsager, then a differentiable iPEPS

Corner transfer matrix renormalization on a corner-and-edge environment. Two problems
share one core: the classical 2D Ising partition function, whose free energy per site has
Onsager's closed form, and a single-site iPEPS whose energy you differentiate.

[`examples/ising2d.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/ising2d.py)
is the Ising half through the library, on a core install; its output is committed on the
[2D Ising CTMRG](../examples/ising2d.md) page.
[`examples/toy_codes/ctmrg.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/toy_codes/ctmrg.py)
writes a C4v CTMRG out on the tensor layer and adds the iPEPS gradient; it needs the
`jax` extra.

[`examples/ising_thermo.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/ising_thermo.py)
differentiates the library lane's contraction twice; its output is on
[2D Ising thermodynamics by AD](../examples/ising-thermo.md).

```sh
uv run python examples/ising2d.py
uv run --extra jax python examples/ising_thermo.py
uv run --extra jax python examples/toy_codes/ctmrg.py
```

The blocks below are illustrative rather than doctests: a converging environment sweep at
`max_bond=24` runs for a hundred sweeps, and the gradient half needs the `jax` extra.
Every printed number quoted on this page is from the committed output of one of those
files.

## The two problems, stated

**Classical Ising.** $H = -\sum_{\langle ij\rangle} s_i s_j$ on the square lattice with
$s_i = \pm 1$, and

$$
Z(\beta) = \sum_{\{s\}} e^{-\beta H},
\qquad
\beta f = -\lim_{N\to\infty}\frac{1}{N}\ln Z .
$$

$Z$ is already a tensor network. Split each bond weight symmetrically,
$e^{\beta s s'} = \sum_\mu W_{s\mu}W_{s'\mu}$ with

$$
W = \begin{pmatrix}\sqrt{\cosh\beta} & \sqrt{\sinh\beta}\\[2pt]
                   \sqrt{\cosh\beta} & -\sqrt{\sinh\beta}\end{pmatrix},
$$

and sum out each site spin, leaving one rank-4 tensor per site:

$$
a_{tlbr}(\beta) = \sum_{s=\pm 1} W_{st}W_{sl}W_{sb}W_{sr},
\qquad
Z = \operatorname{tTr}\bigotimes_{\text{sites}} a .
$$

The columns of $W$ are the $\mathbb{Z}_2$ parity basis — column $0$ is even in $s$,
column $1$ odd — so the sum over $s$ annihilates every entry whose four leg parities
multiply to odd. Those eight entries have no block in a graded `SymmetricTensor`; the
grading *is* the model, not a claim checked afterwards.

**iPEPS.** A single site tensor $A^{s}_{lurd}$ tiles the plane into a state
$\lvert\Psi(A)\rangle$, and the objective is

$$
E(A) = \frac{\langle\Psi(A)\vert H\vert\Psi(A)\rangle}
             {\langle\Psi(A)\vert\Psi(A)\rangle}.
$$

Both quantities are contractions of an infinite network, and CTMRG is how both are
approximated.

## What the environment approximates

CTMRG replaces the infinite lattice around a patch by a corner tensor $C$ and an edge
tensor $T$: $C$ stands for a whole quadrant of the lattice, $T$ for a half-row, each
compressed onto an environment bond of dimension $\chi$. The sweep alternates *absorb* —
grow $C$ and $T$ by one row/column of bulk tensors — and *renormalize* — project the
grown object back onto $\chi$ states with an isometry read off an SVD. The fixed point of
that map is the environment.

For the classical model the observable then comes out by Baxter's corner-transfer
telescoping,

$$
\kappa = \frac{Z_{(L+1)\times(L+1)}\; Z_{L\times L}}
              {Z_{(L+1)\times L}\; Z_{L\times(L+1)}},
\qquad
\ln \kappa = \frac{1}{N}\ln Z,
$$

since $(L+1)^2 + L^2 - 2L(L+1) = 1$: three patches built from the same $C$ and $T$
differ by exactly one site, so every environment tensor and every gauge factor cancels.
That ratio is `log_kappa` in the example files, and $\beta f = -\ln\kappa$.

## The two lanes

[`EnvCTM`][tenet.network.EnvCTM] is the directional environment: four corners and four
edges per unique site ([`EnvLocal`][tenet.network.EnvLocal]), and the moves `'l'`, `'r'`,
`'t'`, `'b'` — or `'h'` and `'v'`, which update every site at once. It takes a
[`Peps`][tenet.network.Peps] and asks nothing of it beyond the lattice.

[`EnvCTMc4v`][tenet.network.EnvCTMc4v] is its C4v specialization: one corner and one edge
([`EnvLocalC4v`][tenet.network.EnvLocalC4v]) read under all eight names, and the single
move `'d'`. It needs an ansatz with four *identical* virtual legs, because the 90-degree
rotation cycles them and acts only if they are one leg. Four identical legs do not tile
the plane with themselves, so the plane is a checkerboard of `a` and
[`flip`][tenet.network.flip]`(a)` — every leg's `side` reversed, every block kept — and
the odd sublattice's corner and edge are the flips of the one pair stored.
[`PepsFlip`][tenet.network.PepsFlip] is the view that hands them back.

## The bulk tensor is the physics, and it is yours

```python
c, s = np.sqrt(np.cosh(beta)), np.sqrt(np.sinh(beta))
w = np.array([[c, s], [c, -s]])
block = np.einsum("st,sl,sb,sr->tlbr", w, w, w, w)
```

$W W^{\mathsf{T}}$ is the bond Boltzmann weight $\begin{pmatrix} e^{\beta} & e^{-\beta} \\
e^{-\beta} & e^{\beta}\end{pmatrix}$, split symmetrically so half a bond sits on each of
the two sites it joins. Then `s` is the site spin, summed over, and `t`, `l`, `b`, `r` are
the four half-bonds leaving it — so the einsum is literally "one Ising spin with four
legs".

```python
space = GradedSpace.new(Z2, {Z2Sector(0): 1, Z2Sector(1): 1})
a = SymmetricTensor.from_dense(block, (Leg(space, OUT),) * 4)
```

That `W` *is* the parity basis, so the Z2 grading is a statement rather than a claim
checked afterwards: summing over `s` annihilates every entry with an odd number of odd
legs, and those eight entries have no block to live in — `from_dense` would have raised
had one been non-zero. All four legs are `OUT` and identical: a leg direction only matters
where two legs are joined, and the C4v lane contracts this tensor against itself.

## The converging sweep

```python
env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), a))
out = env.iterate_(max_bond=24, corner_tol=1e-10)
```

A `1x1` unit cell: the Boltzmann tensor is the same on every site, so one tensor tiles the
lattice and one corner/edge pair is its whole environment.
[`update_`][tenet.network.EnvCTM.update_] is one sweep and
[`iterate_`][tenet.network.EnvCTM.iterate_] the loop, sweeping until the corner spectra
stop moving within `corner_tol` and reporting a [`CTMRG_out`][tenet.network.CTMRG_out] —
`sweeps`, `max_dsv`, `converged`.

```python
env[0, 0].tl, env[0, 0].t          # one corner, one edge
```

`max_bond` is an input; the realized environment bond is on the corner's own legs, which
is where you read what the run actually kept. `init='eye'` (the default) seeds a
one-dimensional environment bond; `init='dl'` absorbs one layer of the network into that
seed without truncating.

A rank-4 site tensor, like the Boltzmann tensor above, is a classical partition function
and is used as it is. A rank-5 one — an iPEPS, with a physical leg — becomes a
[`Peps2Layers`][tenet.network.Peps2Layers] view whose items are lazy
[`DoublePepsTensor`][tenet.network.DoublePepsTensor] pairs: the bra-ket product is never
formed, and the environment edge carries the ket bond and the bra bond adjacent and
separate.

At `max_bond=24` the Ising half reproduces Onsager's free energy to float precision off
criticality — the committed output records a relative error of `6.7e-16` at $\beta = 0.3$
and `8.9e-16` at $\beta = 0.5$ — and `2.2e-6` at $\beta_c$, where the correlation length
outruns any finite environment.

## Closing the boundary: what is traced against what

The environment is a boundary, not a number. Turning it into the partition function per
site is Baxter's corner-transfer telescoping, $\kappa = z_a z_c / z_h^2$, and it is the
example's job rather than the library's:

```python
c, cf, t, tf = e.tl, flip(e.tl), e.t, flip(e.t)
z_c = tenet.full_trace(tenet.einsum("ab,ac,dc,eb->de", c, cf, c, cf))
```

Four corners in a ring, each sharing one leg with the next, and
[`full_trace`][tenet.full_trace] closing the ring: that is $Z$ on an $L \times L$ patch.
Every contracted pair has to meet `IN` against `OUT`, and a ring of four corners changes
sublattice twice, so two of the four enter as `flip(e.tl)` — same blocks, reversed leg
directions.

```python
z_h = tenet.full_trace(tenet.einsum("ab,ac,dfc,ed,eg,gfh->hb", c, cf, tf, cf, c, t))
z_a = tenet.full_trace(
    tenet.einsum("ab,apc,cd,eqd,fe,grf,gh,hsk,spqr->kb", c, t, c, t, c, t, c, t, a)
)
```

`z_h` wedges two opposite edges between the four corners — an $L \times (L+1)$ patch, with
`f` the physical bond the two facing edges share. `z_a` rings a corner, an edge, a corner,
an edge, … around one bulk tensor, an $(L+1) \times (L+1)$ patch; `p`, `q`, `r`, `s` are
the four half-bonds the edges hand to `a`. Nothing is flipped in `z_a`, because corners
and edges already alternate sublattice around the ring.

The three patches telescope: $(L+1)^2 + L^2 - 2L(L+1) = 1$, so every environment tensor
cancels and exactly one site's worth of partition function is left — gauge and
normalization included, which is why no separate normalization step appears anywhere.

## The projectors assume nothing

Under `EnvCTM` a projector pair comes from the QR of each half of the 4x4 patch and an SVD
of `r0 @ r1ᵀ` ([`proj_corners`][tenet.network.proj_corners]): the two sides enter as two
tensors and leave as two projectors. Under `EnvCTMc4v` the projector is the `U` of an SVD
of the 2x2 enlarged corner ([`corner2x2`][tenet.network.corner2x2]) and the renormalized
corner is `V† U S`, so the two index groups leave as two different factors and the
correction between them is kept.

Neither construction needs the enlarged corner to be Hermitian. That is a property of the
*ansatz* — it holds exactly when the state carries the full C4v point group, all four
rotations and all four reflections — so an algorithm that needs it is an algorithm with a
precondition it cannot check.

Reading the corner spectrum is an SVD for the same reason:

```python
spectrum(tenet.linalg.svd(env[0, 0].tl, ((0,), (1,)))[1])
```

The renormalized corner is `V† U S`, not a diagonal matrix, so the singular values are not
sitting on a diagonal waiting to be read — the `((0,), (1,))` axis grouping says which
index group is the map's codomain and which its domain, and `svd` produces them.

## Grading the Ising half by Z2

The Boltzmann bulk tensor is Z2-graded. That stops a finite-`max_bond` environment from
breaking the symmetry spuriously in the ordered phase, which is what lets the run go above
$\beta_c$ against Onsager at all. Two further things the grading gives:

- **Zero magnetization is structural.** A spin insertion is a Z2-odd tensor, which no
  invariant `SymmetricTensor` can hold, so `from_dense` refuses it. The refusal is the
  statement.
- **The ordered-phase corner spectrum is exactly two-fold degenerate**, the doublet
  spanning the two parity sectors. The committed output, at $\beta = 0.5$:

```text
corner spectrum at beta=0.5: 0.6905 0.6905 0.1486 0.1486 0.0320 0.0320
```

Below $T_c$ the two ordered states are degenerate, and each singular value appears once in
the even and once in the odd Z2 sector. Because that doubling is *cross*-sector, and the
broadened SVD rule in `tenet.ad` broadens *per coupled sector*, a graded run never hands
one SVD a degenerate pair.

## The differentiable form

`iterate_` runs outside `jit` and `grad` and cannot be otherwise: it loops on a measured
spectrum change and re-decides the environment bond each sweep.
[`EnvCTMc4v.update_`][tenet.network.EnvCTMc4v.update_] with `bond=` is the traceable form
— a fixed-structure move on a bond you decided outside.

```python
import jax
import tenet

tenet.enable_jax(ad=True)          # pytrees + the broadened SVD/eigh VJPs
```

`enable_jax` registers `SymmetricTensor` as a pytree; `ad=True` additionally replaces
JAX's SVD and eigh VJPs with the Lorentzian-broadened form, which is what keeps a
degenerate environment spectrum from producing `NaN`.

```python
warm = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), ising_bulk(beta)))
warm.iterate_(max_bond=chi)                       # decided outside
bond = warm[0, 0].tl.legs[0].space
seed = (warm[0, 0].tl, warm[0, 0].t)
```

Converge once, untraced, and keep two things out of it: `bond`, the environment
`GradedSpace` the sweep settled on, and `seed`, the converged corner and edge to start the
traced moves from. Warming up outside is what makes four unrolled moves enough inside.

```python
def beta_f(b):
    env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), ising_bulk(b)), init=None)
    env.env[0, 0].tl, env.env[0, 0].t = seed
    for _ in range(4):
        env.update_(bond=bond)                    # projected inside
    return -log_kappa(env)

jax.grad(beta_f)(0.4)
```

`init=None` builds no seed environment, because `seed` is written in directly on the next
line. The four `update_` calls are a fixed number of fixed-structure moves — no loop
condition on a measured value, no bond re-decided — which is exactly what `grad` can
traverse.

That is the same decide-outside / project-inside pairing as
[truncation](../guide/truncation.md): `svd_truncated` chooses a bond space out here,
`svd(..., bond=)` reuses it in there. `bond` is a `GradedSpace` — frozen, hashable,
array-free metadata — so it enters `jit` as a cache key rather than as a flattened leaf.

The gradient is a **truncated backprop through `k` unrolled moves**, not the implicit
fixed point. The two are different objects: the implicit derivative differentiates the
fixed-point *condition* $C = \mathcal{F}(C, a(\beta))$ and needs a linear solve inside the
VJP; this differentiates a composition of $k$ concrete moves whose initial condition is a
constant. They agree in the limit $k \to \infty$, and the difference at finite $k$ is
measurable rather than a caveat.

Its oracle is Onsager's closed form, twice over:

$$
u = \frac{\partial(\beta f)}{\partial\beta},
\qquad
c_V = -\beta^{2}\frac{\partial^{2}(\beta f)}{\partial\beta^{2}} .
$$

`tests/integration/test_ctmrg.py` checks $u$ against $\mathrm{d}(\beta f)/\mathrm{d}\beta$,
and [2D Ising thermodynamics by AD](../examples/ising-thermo.md) takes the second
derivative as well — where the finite $k$ becomes visible, because the environment sits at
its fixed point when the traced region starts, so $u$ is converged at $k = 2$ while $c_V$
is still moving at $k = 8$.

## The iPEPS half

The variational chain is

$$
A \;\rightarrow\; \lvert\Psi(A)\rangle \;\rightarrow\; \{C, T\}
\;\rightarrow\; E(A) \;\rightarrow\; \nabla_A E \;\rightarrow\; A' = A - \eta\,\nabla_A E ,
$$

with $E(A)$ evaluated as a ratio of two contractions of the same $2\times 1$ patch — the
numerator with the two sites' physical legs held open for $h$ to close, the denominator
with them closed against each other. The environment is only defined up to a scale, so
the *ratio* is the observable and the numerator alone is not.

`jax.grad` differentiates that whole chain, including the environment: the same
decide-outside / project-inside pairing as the Ising half, with the converged $(C,T)$ as
a constant initial condition and $k$ unrolled moves carrying the derivative. The gradient
is a `SymmetricTensor` with $A$'s own structure, so `jax.tree.map(lambda p, g: p - lr*g,
a, grad)` touches only block values and the updated ansatz is symmetric by construction —
no projection back onto the symmetric manifold, because the step never left it.

A single-site iPEPS over a **self-conjugate** virtual space, averaged over the eight
elements of C4v, with a random symmetric two-site `h`. It exercises graded truncation,
`svd(bond=)` across sectors and multiplet degeneracies, and both lanes measure the same
number on it: `EnvCTMc4v` on a one-site cell, and `EnvCTM` on a
[`CheckerboardLattice`][tenet.network.CheckerboardLattice] of `a` and `flip(a)`.

A rotation identifies the virtual space with its dual, so the point group is available
only on a self-conjugate space. Every SU(2) space is one; a U(1) space such as
`{q = 0: 1, q = +1: 1}` is not, and no ansatz on it carries a rotation at all — `EnvCTM`
is the lane for those, because it has no point group to require.

It makes **no benchmark-energy claim**, and a one-site unit cell cannot. A single-site AFM
Heisenberg cell needs one sublattice rotated by $\pi$ about $y$, which turns
$S^x S^x - S^y S^y$ into $(S^+ S^+ + S^- S^-)/2$ — an operator that changes
$S^z_{\mathrm{tot}}$ by $\pm 2$ and so destroys the U(1) the ansatz is graded by.

## What the library owns

`tenet.network` owns the environment: the seeds, the moves, the sweep, the truncation, the
traced form. What it does not own, and the example supplies: the bulk tensor, the ansatz —
a library that symmetrized your input would be silently editing your state — and what to
measure, which is a geometry-specific reduced density matrix. The telescoping above is in
that second list for the same reason: which patches to trace against which is a property
of the lattice you are on.

## Where next

- [JAX and backends](../guide/jax-and-backends.md) — `enable_jax(ad=True)` and what the
  broadened VJP assumes.
- [Truncation](../guide/truncation.md) — the pairing this page leans on.
- [`tenet.network`](../api/network.md) — `EnvCTM`, `EnvCTMc4v`, `corner2x2`,
  `proj_corners`, `flip`.
