# Saving and loading

## One tensor

[`tenet.save`][tenet.save] writes a single `.npz`: a JSON header describing the legs,
plus one array per block. [`tenet.load`][tenet.load] reads it back.

```python
>>> import os, tempfile
>>> import tenet
>>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
>>> from tenet.symmetry import U1, U1Sector
>>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
>>> t = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
>>> with tempfile.TemporaryDirectory() as d:
...     tenet.save(t, os.path.join(d, "t.npz"))
...     t2 = tenet.load(os.path.join(d, "t.npz"))
>>> t2.structure == t.structure
True
>>> bool(tenet.allclose(t2, t))
True

```

The structure comes back exactly equal, so a loaded tensor is element-wise compatible
with the one you saved.

`t.save(path)` and `SymmetricTensor.load(path)` are the same two functions as methods.

## What a saved file guarantees

- **Blocks come back as NumPy.** A device placement is not a property of the tensor, so
  the restore is `load(path).to_backend("jax")`. Any backend saves: blocks are converted
  with `autoray`'s `to_numpy` on the way out.
- **The gauge is checked.** For SU(2), SU(N) and fermionic parity, the file carries a
  fingerprint of the recoupling conventions that produced its coefficients, and `load`
  verifies it. Block coefficients are meaningful only against those conventions, so a
  gauge-mismatched file is refused with a `ValueError`.
- **`compress=False` is the default.** Reduced blocks are dense float arrays that do not
  compress well, so paying `zlib` on every checkpoint buys nothing. Pass `compress=True`
  if you want it. The zip container costs a constant overhead: a 4-block SU(2) tensor
  with 504 bytes of block data writes a 2282-byte file.

`save` refuses a `Leg` whose `name` is not `None`, `str` or `int`, before writing
anything, naming the public axis; and it refuses a leg whose provider is not one of the
serializable kinds. `load` raises for a future format version, a header block count
contradicting the structure, or a member set that is not the header plus `b0..b{n-1}`.

## A whole MPS

[`MPS.save`][tenet.network.MPS.save] writes a **directory**: one `NNN.npz` per site plus
`mps.json`.

```python
>>> import tempfile
>>> from tenet.models import spin_half
>>> from tenet.network import MPS
>>> site = spin_half()
>>> psi = MPS.product(site.phys, [U1Sector(1), U1Sector(-1)] * 3)
>>> with tempfile.TemporaryDirectory() as d:
...     path = os.path.join(d, "state")
...     psi.save(path)
...     phi = MPS.load(path)
>>> len(phi) == len(psi)
True

```

It goes through `tenet.save` per site, which keeps the gauge verification on the path for
every tensor in the chain. The container is a directory because `np.load` reads a flat
`.npz` and does not descend into a nested one.

`MPS.save` raises `FileExistsError` if the directory exists and is not empty, **before
anything is written**. Sites load as NumPy; restore a device with `.to_backend("jax")`
per site.

## Checkpointing a DMRG run

Saving the state is all a restart needs. The schedule is a list, so re-entering it is a
slice:

```python
out = dmrg_(psi, h, schedule=schedule[:2])
out.psi.save("checkpoint")
# ... later, in another process
psi = MPS.load("checkpoint")
out = dmrg_(psi, h, schedule=schedule[2:])
```

A sweep is a full round trip, so there is no direction to restore, and the slice is the
position. See [DMRG](dmrg.md).

## Where next

- [`tenet.serialize`](../api/serialize.md) — the reference.
- [Symmetries and providers](symmetries-and-providers.md) — what a gauge fingerprint is.
