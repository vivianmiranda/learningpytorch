---
name: py-module-style-conventions
description: "Code style for the emulator/ PACKAGE and driver/ CLI files -- DISTINCT from the read-only teaching NOTEBOOK pytorch1.ipynb (which keeps the slide rules: <=~60 cols + hanging indent, [[hanging-indent-not-paren-alignment]]). Set by the user this session (2026-06-29): (1) NAMED PARAMETERS everywhere the callee allows -- the user's words, 'I will forget the meaning of position X'; covers our functions/methods, cls(...) constructors, nn.Linear/Conv1d (which is in vs out?), library config kwargs (dtype=/device=/dim=/color=), torch.cat(dim=), Normalize(vmin=/vmax=). (2) IRREDUCIBLY-POSITIONAL args (matplotlib plot/semilogy x/y data, torch.einsum operands, the array/tensor SUBJECT of torch.cat/from_numpy/np.asarray/np.zeros, a module call model(x)) STAY positional AND get a COMMENT naming them. (3) PAREN-ALIGNMENT (one arg per line under the opening paren) for .py -- OVERRIDES the hanging-indent slide rule. (4) 90-COLUMN width for .py (relaxed from slide ~60). (5) DIDACTIC comments on tricky mechanics, ESPECIALLY tensor-shape ops (unsqueeze/expand/[:,None] broadcasting, view, np.searchsorted); flag the geometry's OWN .unsqueeze (scatter-to-full-vector) vs torch.unsqueeze (add a size-1 axis) name-clash. GOTCHA: keep *args forwarders POSITIONAL (keywording pred= before *args -> 'multiple values' bug). DOC QUALITY (2026-06-30): module docstrings are PROSE (real subjects, not slash-as-subject lists); PS: jargon defs at file end (whitened/encoded/resident/loader/dump/memmap/Mahalanobis); no double-dash; blank-line-group dense method bodies (leave aligned = tables); cross-module call sites get a (module.py): what-it-does provenance comment; enumerate EVERY block-dict key in the docstring; comment math as an explicit display formula with named symbols."
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

## Documentation quality (ADDED 2026-06-30, the user reviewed the package docs)

The user read the docstrings/comments and pushed hard on readability "because
students will read the documentation". Six rules:

6. **Module docstrings are PROSE, not telegraphic notes.** A short paragraph of
   real sentences (subject + verb): "This module is the output side: it owns
   ...", "These are the full networks that map ...". NEVER a noun-phrase fragment
   ("The output side: own ...") and NEVER slash-as-subject ("A / B / C compute
   ...") -- write "The functions A, B, and C compute ..." / "A and B score the
   model". (slash-as-subject was the recurring smell across the package.)

7. **Define jargon in-file with a `PS:` at the END of the module docstring.** The
   user flagged whitened / encoded / resident as opaque; per file that USES such a
   term, add `PS: whitened = ...` (also encoded / resident / loader / dump /
   memmap / Mahalanobis / squeeze) so a reader of THAT file alone gets it.
   Canonical defs: whitened = rotated into the covariance eigenbasis + scaled to
   unit variance (decorrelated, equally hard to fit); encoded = a dv put through
   the geometry's encode (kept entries, centered, whitened); resident = held in
   GPU memory the whole run, not re-loaded each batch; loader = a closure
   load(rows) -> tensor mapping global row indices to a ready-to-train batch on
   the device, hiding where the data lives (resident on GPU / streamed from RAM /
   read from a disk memmap); dump = the full on-disk array written by the
   data-generation run, one row per cosmology (the dv dump is the .npy, the param
   dump the .txt), from which a run draws its N_train subset.

8. **No `--` (double dash).** The user dislikes it: use commas, colons, parens, or
   "i.e." (docstrings and comments; "eliminate most", do not obsess).

9. **Jump lines: blank-line-group dense method bodies** into read / build /
   compute / return paragraphs ("you never jump lines ... very hard for a human to
   read"). Find the WALLS with a quick ast pass (functions whose body has a long
   run of zero blank lines); leave an ALIGNED `=` assignment table intact (the
   alignment is the readability device).

10. **Cross-module call sites get a provenance comment** naming WHERE the function
    lives and WHAT it does: `# load_source (data_staging.py): memmap the dv, cut,
    stage one source` -- so a reader of one file follows the pipeline without
    opening others.

11. **A block (dict) parameter's docstring enumerates EVERY key**, never
    "settings": e.g. the `data` block lists train_dv / train_params / train_covmat
    / val_dv / val_params / cosmolike_* / omegabh2_cut / train_divisor /
    val_divisor / split_seed / ram_frac; `train_args` lists nepochs / bs /
    loss_mode / silent + the six sub-blocks (model / optimizer / lr / scheduler /
    trim / focus) and their keys.

12. **Comment math as an explicit display formula with named symbols** -- the user
    called `pred = base * (1 + net_output)` (then defining base / net_output)
    "already more didactics" than prose; put the equation on its own `#   ...` line.

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
