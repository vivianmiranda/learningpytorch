---
name: py-module-style-conventions
description: "Code style for the emulator/ PACKAGE and driver/ CLI files -- DISTINCT from the read-only teaching NOTEBOOK pytorch1.ipynb (which keeps the slide rules: <=~60 cols + hanging indent, [[hanging-indent-not-paren-alignment]]). Set by the user this session (2026-06-29): (1) NAMED PARAMETERS everywhere the callee allows -- the user's words, 'I will forget the meaning of position X'; covers our functions/methods, cls(...) constructors, nn.Linear/Conv1d (which is in vs out?), library config kwargs (dtype=/device=/dim=/color=), torch.cat(dim=), Normalize(vmin=/vmax=). (2) IRREDUCIBLY-POSITIONAL args (matplotlib plot/semilogy x/y data, torch.einsum operands, the array/tensor SUBJECT of torch.cat/from_numpy/np.asarray/np.zeros, a module call model(x)) STAY positional AND get a COMMENT naming them. (3) PAREN-ALIGNMENT (one arg per line under the opening paren) for .py -- OVERRIDES the hanging-indent slide rule. (4) 90-COLUMN width for .py (relaxed from slide ~60). (5) DIDACTIC comments on tricky mechanics, ESPECIALLY tensor-shape ops (unsqueeze/expand/[:,None] broadcasting, view, np.searchsorted); flag the geometry's OWN .unsqueeze (scatter-to-full-vector) vs torch.unsqueeze (add a size-1 axis) name-clash. GOTCHA: keep *args forwarders POSITIONAL (keywording pred= before *args -> 'multiple values' bug)."
metadata:
  node_type: memory
  type: feedback
---

Style for the **ported .py package** (`emulator/`) and the **CLI drivers**
(`driver/`). This is DISTINCT from pytorch1.ipynb, which is read-only and keeps
the slide rules (<=~60 cols, hanging indent, [[hanging-indent-not-paren-alignment]],
[[emulator-pipeline-and-goal]]). The user set these for the .py code this
session, while we translated the notebook ([[emulator-python-package]]).

1. **NAMED PARAMETERS everywhere the callee allows.** The user's reason: "I will
   forget the meaning of position X." Name args to our functions/methods,
   `cls(...)` constructors, `nn.Linear`/`Conv1d` (`in_features=`/`out_features=`
   -- which is in vs out?), library config kwargs (`dtype=`/`device=`/`dim=`/
   `color=`), `torch.cat(dim=)`, `Normalize(vmin=/vmax=)`.

2. **Irreducibly-positional args stay positional AND get a naming comment.**
   These callees REJECT keywords for their data: matplotlib `plot`/`semilogy`
   x/y; `torch.einsum` operands; the array/tensor SUBJECT of `torch.cat` /
   `from_numpy` / `np.asarray` / `np.zeros`; a module call `model(x)`. Comment
   forms in use: `# x = epochs, y = train loss` (matplotlib) and
   `# operands in subscript order: r (b,i)=residual, Cinv (i,j), r (b,j)`
   (einsum).

3. **Paren-alignment** (one argument per line, aligned under the opening paren)
   for .py files -- this OVERRIDES the hanging-indent slide rule
   ([[hanging-indent-not-paren-alignment]]) for `emulator/` and `driver/`.

4. **90-column width** for .py (relaxed from the slide ~60).

5. **Didactic comments on tricky mechanics**, ESPECIALLY tensor-shape ops:
   `unsqueeze` (insert a size-1 axis at `dim`, for broadcasting), `expand`
   (stride-0 VIEW, no copy), `[:, None]` / `[None, :]` (the numpy/torch spelling
   of `unsqueeze`), `view` reshape, `np.searchsorted` (global->local row remap).
   NAME-CLASH to call out every time it appears: the geometry's OWN `.unsqueeze`
   (scatter the kept entries into a full zero vector) vs torch's
   `tensor.unsqueeze` (add a size-1 axis) -- the user specifically struggles
   with this.

**Gotcha:** keep `*args` FORWARDERS positional. Keywording `pred=`/`target=`
BEFORE a `*args` (e.g. `super().loss(pred, target, *args, **kwargs)`) makes a
non-empty `*args` refill `pred`'s slot -> "multiple values for 'pred'". Exclude
any call that carries a Starred positional from the keywording pass.

**Why:** the user is a cosmology expert and a Python learner; explicit parameter
names plus shape-op comments are what make the ported code readable and
maintainable to THEM (the same teaching impulse as the notebook, applied to real
code). **How to apply:** when writing or editing .py in `emulator/` or
`driver/`, name every keyword-able arg, comment the irreducible positionals,
paren-align wrapped calls at 90 cols, and add a few lines explaining any
non-obvious reshape/broadcast. Validate the keyword names mechanically against
the real signatures (a quick ast pass) -- a wrong name compiles but breaks at
call time.
