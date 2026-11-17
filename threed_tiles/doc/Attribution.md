# Persistent 3D Tiles attribution

Direct Google, Cesium ion and Custom imports share the same attribution
registry and display. New imports add credits rather than replacing earlier
imports' credits.

`scene.blosm.importAttributions` is a persistent `CollectionProperty`. Each
`BlosmImportAttribution` entry contains a `collection` pointer and native custom
properties:

- `copyrights`: unique tile copyright entries, split at semicolons as before.
- `counts`: occurrence counts corresponding to those entries, used for sorting.
- `providerCredits`: additional attribution strings returned by Cesium ion.

These values are saved in the blend file. String lists are stored directly;
no JSON encoding or collection names are needed.

## Display and collection lifetime

The Copyright panel calls `get_copyright(scene.blosm)` directly. This read-only
formatter builds a combined message from live registry
entries, sorts tile notices by their accumulated occurrence counts, and removes
exact duplicate notices/provider credits. Different notices from the same
holder remain distinct. The formatter and panel callbacks never write a display
cache or remove entries.

Validity is based only on the stored collection reference. There is no scene
hierarchy traversal, global collection search, timer or dependency-graph
handler for attribution. Deleted collections are excluded on the next read.
Renamed, hidden, empty and unlinked collections continue contributing while
their references remain valid. A new collection with a deleted collection's
former name does not inherit its credits.

Old blend files can contain a stored scene-level `copyright` value without any
association to imported collections. The formatter preserves this legacy text as
an additional contribution and deduplicates it against new entries. Its
collection ownership cannot be reconstructed. The legacy value is read through
`scene.blosm.get("copyright", "")`; there is no registered `copyright` RNA
property or setter. New imports never copy computed display text into this
legacy storage.

## Import commits and undo

`record_import()` runs only after geometry has been successfully committed. It
removes entries with invalid collection references, then creates or updates the
entry for the import's permanent collection. Cleanup belongs to the import's
existing undo step. Registry entries may therefore remain stored after deletion
until the next successful import, while their credits are already hidden.

The blocking renderer publishes after finalizing retained geometry. The
progressive renderer publishes after each successful tile commit, updating the
same destination entry. Its temporary per-tile renderers collect credits but
do not publish entries. Cancellation retains credits for retained geometry;
failed tiles contribute no credits. An import producing no geometry does not
create an entry or clean existing entries.

## Verification

- `tests/check_attribution.py`: mixed Google/ion/Custom imports, reference-only
  aggregation, duplicates, legacy text, deletion/rename/unlink behavior,
  failures/cancellation, native string-list and pointer save/reopen persistence,
  and independent scene registries.
- `tests/check_attribution_ui.py`: real read-only panel callbacks, deletion
  filtering and Undo/Redo, cleanup within the next import undo step, partial
  progressive cancellation, and addon disable/enable registration.

Run these checks with factory startup, disabled automatic execution and an
isolated user profile. Generated blend files and reports belong outside the
checkout in the chat's designated artifact directory.
