# Reproducibility

Record the software Git commit, `SOURCE_MANIFEST.json`, dependency versions, effective configuration and input/output hashes with each study. The source manifest binds the geometry implementation's file inventory and identity; runtime input checks reject mismatched source or geometry status records.

Placement and size/shape seeds are independently configurable through `--seed` and `--size-seed`. An omitted size seed inherits the effective placement seed. Retain the requested and effective seeds, candidate sequence and size-generation replacements with your outputs. `--realizations 1` requests one accepted geometry and may use several candidates within the bounded retry controller.

Install dependencies from the public sources in [INSTALLATION.md](INSTALLATION.md). Compiler, library, operating-system and Abaqus differences may affect numerical paths. For user-generated geometry, preserve the raw JSON and its adjacent acceptance record before FE construction. For the packaged geometry input, preserve its exact file identity together with the software version. Keep the resulting CAE, particle work JSON and construction status record together. If solution is explicitly enabled, also retain its `run_info.json` and ODB.

Geometry acceptance, CAE construction and solver completion are separate outcomes. Acceptance of one stage does not establish completion of the next. Keep scientific outputs and study records outside the repository.
