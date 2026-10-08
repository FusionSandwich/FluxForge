# Irradiation history inputs

Open **Tools → Irradiation History**. Add intervals with Start and Stop in elapsed
seconds from a common zero and a relative power for each interval. A power of
1 means your chosen reference power; 0 explicitly records a shutdown. Values
above 1 are allowed. The reference must be the same for all rows.

Rows must be chronological, nonoverlapping, and have Stop greater than Start.
All three values must be finite; times and powers cannot be negative. Blank
power is an error. The editor does not assume missing irradiation measurements.

Leading and intermediate gaps are retained as zero-power duration segments so
that decay during shutdown remains in the history. The final Stop defines the
end of irradiation. Include a final zero-power interval if the history needs to
extend beyond the last powered interval; do not also count that period as
post-irradiation cooling in an activity calculation.

**Apply history** validates the current rows and supplies `IrradiationSegment`
objects to the current GUI window. Closing and reopening the dialog retains
rows and the last applied history for that window. Workspace session persistence
does not yet include this history; export JSON to retain it between launches.

**Export JSON** writes the original timeline plus `irradiation.segments` in the
duration/power shape accepted by activation workflows. **Load JSON** restores
an editor export and checks that its duration segments agree with its timeline.
An invalid import leaves current rows intact. An invalid export does not replace
an existing file.

Entered timing does not supply timing/power uncertainty or certify a reactor
operating record. Exports identify their source as user-entered and retain
`scientific_admission: false`; they do not replace qualified operating-history
source evidence.
