# Neptune Exporter Project - Data Model Challenges

## MLflow Model

- **organisation/project** → such division does not seem to exist
  - we can just add it to the experiment name
  - or ask user for an (optional) prefix for each project (as a cli param) and import runs as they are in the data loading step

- **experiment** → experiment
- **run** → run

- **forks** → nested runs
  - mlflow supports nesting runs under a parent. It does not seem to inherit series. The runs are nested in the UI instead.
  - The other option would be to duplicate the data to all forks and save them as independent runs, but it could generate much more data on the mlflow side.

- **attribute path** → attribute key
  - mlflow key constraints: This string may only contain alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/). All backend stores will support keys up to length 250, but some may support larger keys.
  - neptune path constraints: MAX_ATTRIBUTE_PATH_LENGTH = 1024
  - neptune accepts much longer paths, especially if we additionally attempt to prepend them with the project... could attempt to contract the name and hope that users rarely use paths longer than 250.

- **configs** → params
  - neptune: float, int, string, bool, datetime
  - mlflow: all values are stringified
  - Since all values are stringified, uploading our values won't be a problem. Recreating our model from MLflow would be though.

- **tags** → tags
  - mlflow has tags. Use mlflow.set_tag or set on start_run

- **metrics (float series)** → metrics
  - neptune uses steps - decimal(18, 6)
  - mlflow only accepts steps as ints
  - It would be generally possible to transform our decimals into ints with a simple *1_000_000, but if someone uses less precision they may not be satisfied with such a result and would prefer a different transformation

- **files** → artifacts
  - It seems possible to simply upload files from a given local path, not sure if there are some limits that are different, probably depends on the file storage behind both systems

- **file series** → artifacts?
  - mlflow does not allow to save a file per step. I guess we could append our step to the artifact name.

- **string series** → artifacts?
  - mlflow does not have a direct equivalent. It has log_text (Log text as an artifact), but we would have to encode our steps in the string series. Or log_table, which could retain a two-column structure.

- **histogram series** → artifacts?
  - mlflow does not have an equivalent. It's worse in the case of this type, than with other series, because it won't properly display the histograms. Again, we could save them as artifacts using log_table.

## W&B Model

### Implementation Status: ✅ Complete

- **organisation/project** → entity/project
  - **Implementation**: W&B requires an entity (organization/username) specified via `--wandb-entity`
  - Neptune `project_id` and `experiment_name` are combined to create W&B project name
  - Format: `{name_prefix}_{project_id}_{experiment_name}` (sanitized)

- **experiment** → group

- **run** → run
  - **Implementation**: Created using `wandb.init()` with entity, project, and name
  - Run names follow format: `{run_id}`

- **forks** → forks
  - **Implementation**: ✅ W&B native forks using `wandb.init(fork_from=f"{entity}/{project}/{run_id}?_step=0")`
  - Fork relationships are established at run creation time
  - **Limitation**: W&B doesn't support updating fork relationships after run creation

- **attribute path** → metric/config names
  - **W&B constraints**:
    - Allowed characters: Letters (A-Z, a-z), digits (0-9), and underscores (_)
    - Starting character: Names must start with a letter or underscore
    - Pattern: `/^[_a-zA-Z][_a-zA-Z0-9]*$/`
  - **Implementation**: Sanitization replaces invalid characters with underscores, ensures valid start character

- **configs** → config
  - **Neptune types**: float, int, string, bool, datetime, string_set
  - **Implementation**: Native type preservation using `wandb.config.update()`
    - float → float
    - int → int (explicit conversion)
    - string → string
    - bool → bool (explicit conversion)
    - datetime → string (isoformat)
    - string_set → list

- **tags** → tags
  - **Implementation**: Not yet implemented (W&B supports tags via `wandb.init(tags=[...])`)

- **metrics (float series)** → log floats
  - **Implementation**: ✅ Using `wandb.run.log(metrics_dict, step=int_step)`
  - Steps converted from Neptune decimal(18, 6) to W&B integers
  - Multiplier automatically determined based on data precision (e.g., 100 for 2 decimals, 1 for integers)
  - Metrics grouped by step and logged together for efficiency

- **string series** → W&B Tables
  - **Implementation**: ✅ Using `wandb.Table(columns=["step", "value", "timestamp"], data=...)`
  - Much better than text files - provides native visualization and analysis
  - Each string series becomes one table logged to the run

- **histogram series** → W&B Histograms
  - **Implementation**: ✅ Using `wandb.Histogram(np_histogram=(values, edges))`
  - Native W&B histogram objects for proper visualization
  - Logged with step information: `wandb.run.log({attr_name: histogram}, step=step)`

- **files** → artifacts
  - **Implementation**: ✅ Using `wandb.Artifact(name=attr_name, type="file")`
  - Supports both individual files and directories (file_set)
  - Uses `artifact.add_file()` or `artifact.add_dir()` based on path type

- **file series** → artifacts with step
  - **Implementation**: ✅ Step included in artifact name: `{attr_name}_step_{step}`
  - Each step creates a separate artifact for proper organization

- **source code** → artifacts
  - **Implementation**: Treated as regular files/artifacts
  - **Future Enhancement**: Could use `wandb.run.log_code()` for files under `source_code/` path
