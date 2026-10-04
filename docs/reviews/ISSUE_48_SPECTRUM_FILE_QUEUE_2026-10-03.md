# Issue 48: folder queue for summing and conversion

Added individual-file/folder discovery with optional recursion, deterministic
ordering, duplicate-path removal and a background Qt conversion worker.
Append preserves all individual spectra in one native session; Sum checks bins,
calibration, detector and count semantics before combining complete channel
covariance under an explicit independent-acquisition declaration. Missing timing
does not become a misleading partial total. Inputs are hashed before/after reads.

Existing sources and outputs are preserved. The entire queue is read before a
validated session is serialized privately and published by an exclusive atomic
hard link. Failed parsing, incompatible grids and an existing target do not
produce a partial replacement. Hard-link support is a documented filesystem
requirement. All editable controls and the main-window action have stable IDs.

Nine focused model/Qt checks verify discovery, recursive files, deduplication,
independent covariance arithmetic, timing, incompatible inputs, round-trip
uncertainty, worker progress and output preservation. The production action
catalog is verified separately. This work does not change the analysis owner's
raw replay/peak algorithms or the acceptance owner's CSV reader fixes.
