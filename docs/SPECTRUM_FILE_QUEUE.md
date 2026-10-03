# Spectrum summing and conversion

Open **Tools → Spectrum Summing and Conversion**. Add individual raw files or
a folder; **Include subfolders** applies to subsequent folder additions. Only
registered reader extensions are discovered. Duplicate paths are queued once.
Remove selected rows to exclude files before conversion. Unsupported or malformed
files stop the job with a visible error; they are not silently discarded.

Choose **Append separate spectra** to convert the queue into one `.ffs` session
containing every spectrum, with its own bins, calibration, timing, metadata and
channel covariance. This mode supports unlike grids.

Choose **Sum matching spectra** to combine counts into one spectrum. Confirm
that inputs are independent acquisitions; the covariance sum depends on that
declaration and cannot account for unknown cross-acquisition covariance. All
inputs must have identical channel/energy bins and matching detector,
calibration and count-reference/processing metadata. No rebinning or automatic
rescaling is performed. Supplied within-spectrum covariance is retained.

Known live and real times are added separately. If any input lacks a live or
real time, that summed time remains unknown (0), with the availability recorded
in metadata. An acquisition start time is not invented for the combined result.
Source paths, SHA-256 hashes and original timings are retained. The conversion
does not qualify scientific source evidence.

Choose a **new** output path ending in `.ffs` and click **Convert queue**.
Existing output and source files are preserved. The worker reads and validates
the complete queue before publishing the output; a malformed input produces no
output. Atomic publication uses a same-directory hard link and requires a
filesystem that supports hard links. Conversion errors are displayed without
partially replacing a target file. The window stays open while a job finishes.

Open the result using **File → Open Session**. The raw input files are unchanged.
