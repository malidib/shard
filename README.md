# SHARD

This repository contains a Python driver for **SHARD**: a collision-resolving late-stage planet-formation workflow built on `REBOUND`/`mercurius`. The script detects N-body collisions, maps non-special embryo--embryo impacts onto an SPH outcome catalogue, interpolates the largest remnants, reconstructs small debris with mass-weighted KMeans clustering in velocity space, and writes collision/restart diagnostics.

This README describes practical use of the cleaned script:

```text
SHARD_Rebound.py
```

The code is intentionally procedural. There is no command-line interface. To run a different case, edit the **User-facing configuration** block near the top of the script.

---

## 1. What the code does

For each detected collision, the script follows this workflow:

1. Identify the colliding particles and handle special cases.
   - Collisions with the Sun are treated as accretion into the Sun.
   - Collisions with configured gas giants are treated as perfect mergers.
   - Collisions involving test-particle debris are treated as perfect mergers.
2. Compute the six SPH query parameters:
   - impact speed normalized by mutual escape speed, `v0`,
   - impact angle, `alpha`,
   - total mass, `M_tot`,
   - mass ratio proxy, `gamma`,
   - target water fraction, `wft`,
   - projectile water fraction, `wfp`.
3. Interpolate `SPH.table` to obtain the two largest post-impact remnants.
4. Apply survivor-count logic and drop survivors below the minimum resolved fragment mass.
5. Enforce mass and water-budget consistency for survivors and the SPH-derived debris reservoir.
6. Apply `DUST_MASS_RETENTION_FACTOR` to decide what fraction of that debris reservoir is retained as resolved N-body debris.
7. Interpolate neighboring SPH debris catalogues from `SPHDebris_catalogue/`.
8. Cluster the retained debris in velocity space using mass-weighted `KMeans` and normalize the clusters to the retained debris budget.
9. Map survivors and debris from the SPH collision frame back into the global REBOUND frame.
10. Apply immediate energy-based reaccretion checks.
11. Append debris particles, update active/test-particle status, and write diagnostics.

---

## 2. Units and conventions

Unless noted otherwise, the script uses REBOUND units:

| Quantity | Unit/convention |
|---|---|
| Gravitational constant | `G = 1` |
| Distance | AU |
| Mass | Solar masses, `M_sun` |
| Time | `yr / (2*pi)` internally |
| User-facing integration duration | years |
| Checkpoint/output times | years |
| Water fraction, `wf` | percent |
| Orbital angles in input/output snapshots | radians |
| Collision impact angle, `alpha` | degrees |
| `MIN_FRAGMENT_MASS_MEAR` | Earth masses |
| `ACTIVE_DEBRIS_MASS_MSUN` | Solar masses |

Important time conversion:

```python
TIMESTEP_REBOUND = 1.0e-2 * 2.0 * np.pi
```

This corresponds to a physical timestep of `0.01 yr` because REBOUND time is measured in `yr/(2*pi)`.

---

## 3. Expected project layout

The safest way to run the current cleaned script is from the project root with this layout:

```text
project_root/
├── SHARD_Rebound_cleaned.py
├── CONSTANTS.py
├── SPH.table
├── SPHDebris_catalogue/
│   ├── 001_000000.dat
│   ├── 002_000001.dat
│   └── ...
└── scenario_name/
    └── scenario_name.start
```

The default configuration assumes:

```python
PROJECT_ROOT = Path(".")
SCENARIO_NAME = "scenario_name"
START_TAG = "start"
```

With those defaults, the starting file must be:

```text
scenario_name/scenario_name.start
```

For restarts such as:

```python
START_TAG = "t1000000"
```

The script expects:

```text
scenario_name/scenario_name.t1000000
```

### Note about `PROJECT_ROOT`

The cleaned script centralizes `PROJECT_ROOT`, but parts of the original procedural code still use the scenario string as both a directory name and a file-name prefix. For that reason, the most robust usage is:

```bash
cd /path/to/project_root
python SHARD_Rebound_cleaned.py
```

and leave:

```python
PROJECT_ROOT = Path(".")
```

If you want to run from another working directory, test the path construction carefully before launching a long integration.

---

## 4. Python dependencies

The script imports:

```python
numpy
rebound
reboundx
sklearn.cluster.KMeans
```

It also requires a local module:

```python
CONSTANTS.py
```

At minimum, `CONSTANTS.py` must define the symbols used directly by the script:

```python
MEAR    # Earth mass in solar masses
REAR    # Earth radius in AU
SPHRES  # SPH mass-resolution normalization used in interpolation fallbacks
allpar  # SPH grid values for v0, alpha, mtot, gamma, wft, wfp
bases   # mixed-radix encoding bases for SPH catalogue codes
```

The installed `REBOUNDX`/REBOUND configuration must support per-particle parameters named:

```text
wf
code
```

where `wf` stores water fraction in percent and `code` stores the numeric identity label used by the script.

---

## 5. Input files

### 5.1 Starting-condition or restart file

The starting file is loaded with:

```python
start = np.loadtxt(starting_file, usecols=np.arange(0, 9))
hs = np.loadtxt(starting_file, usecols=-1, dtype=str)
sim.N_active = sum(np.loadtxt(starting_file, usecols=-2, dtype=int))
```

Therefore each non-comment row must contain at least these columns:

```text
m  R  a  e  inc  Omega  omega  M  wf  active  code
```

| Column | Meaning | Unit/type |
|---|---|---|
| `m` | particle mass | `M_sun` |
| `R` | particle radius | AU |
| `a` | semimajor axis | AU |
| `e` | eccentricity | dimensionless |
| `inc` | inclination | radians |
| `Omega` | longitude of ascending node | radians |
| `omega` | argument of periapsis | radians |
| `M` | mean anomaly | radians |
| `wf` | bulk water fraction | percent |
| `active` | active-particle flag | integer `0` or `1` |
| `code` | identity label | string |

Example skeleton:

```text
# m[MSUN] R[AU] a[AU] e inc Omega omega M wf[%] active code
1.0 0.00465 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1 SUN
0.0009543 0.000477 5.2 0.05 0.0 0.0 0.0 0.0 0.0 1 JUP
0.0002857 0.000402 9.5 0.05 0.0 0.0 0.0 0.0 0.0 1 SAT
7.50e-8 4.26e-5 0.8 0.1 0.0 0.0 0.0 0.0 10.0 1 E1
```

Supported labels in the current script are:

| Label pattern | Meaning |
|---|---|
| `SUN` | central star |
| `JUP` | Jupiter-like gas giant |
| `SAT` | Saturn-like gas giant |
| `E<number>` | embryo |
| `P<number>` | planetesimal |
| `D<number>` | debris |

If you add more named giant planets or new label classes, update `label_to_code()` and `code_to_label()` before running.

### 5.2 `SPH.table`

`SPH.table` is the tabular SPH outcome catalogue. The script expects one header line followed by one row per SPH collision.

Each row is interpreted as:

```text
id  code  v0  alpha  mtot  gamma  wft  wfp  Nbig  mfr  remnant_1_payload  remnant_2_payload
```

where each remnant payload contains:

```text
r  theta  phi  v  theta_v  phi_v  m_i  W_i
```

Interpretation:

| Field | Meaning |
|---|---|
| `id` | integer SPH collision identifier |
| `code` | six-digit mixed-radix SPH grid code |
| `v0` | impact speed divided by mutual escape speed |
| `alpha` | impact angle in degrees |
| `mtot` | total colliding mass in solar masses |
| `gamma` | smaller/larger mass ratio proxy |
| `wft`, `wfp` | target/projectile water fractions in percent |
| `Nbig` | number of large remnants; `-1` means unavailable/catastrophic entry in this code |
| `mfr` | fragmented mass fraction relative to total mass |
| `m_i` | remnant mass fraction relative to total mass |
| `W_i` | remnant water mass fraction relative to total pre-impact water mass |

### 5.3 `SPHDebris_catalogue/`

Each non-crashed SPH entry should have a corresponding debris file named:

```text
SPHDebris_catalogue/{zero_padded_id}_{code}.dat
```

For example:

```text
SPHDebris_catalogue/007_102111.dat
```

The debris files are loaded with columns:

```text
x  y  z  vx  vy  vz  m  m_over_mtot  wf
```

| Column | Meaning |
|---|---|
| `x, y, z` | SPH-frame Cartesian position in AU |
| `vx, vy, vz` | SPH-frame Cartesian velocity in AU / `yr/(2*pi)` |
| `m` | particle mass in REBOUND units, as provided by the catalogue |
| `m_over_mtot` | particle mass fraction relative to the SPH collision total mass |
| `wf` | particle water fraction in percent |

For each contributing SPH corner, the script discards that corner's first `Nbig` rows and treats the remaining rows as debris samples.

---

## 6. Quick start

### 6.1 Edit the top-of-file configuration

Open `SHARD_Rebound_cleaned.py` and edit only the block labeled:

```python
# ---------------------------------------------------------------------
# User-facing configuration
# ---------------------------------------------------------------------
```

For a short smoke test, use a much shorter integration than the paper default, for example:

```python
TOTAL_INTEGRATION_YEARS = 1.0e3
N_CHECKPOINTS = 10
PRINT_INTERPOLATION_POINTS = False
```

### 6.2 Check syntax

```bash
python -m py_compile SHARD_Rebound_cleaned.py
```

### 6.3 Run

```bash
python SHARD_Rebound_cleaned.py
```

### 6.4 Monitor progress

During a run, inspect:

```text
<scenario>/progress.<scenario>
<scenario>/events.dat
<scenario>/coll.dat
```

The terminal and progress file report current time, particle count, wall-clock timing, and relative energy error.

---

## 7. User-facing configuration reference

The following values are intended to be edited by users. They are defined near the top of `SHARD_Rebound_cleaned.py`.

### 7.1 Paths and scenario selection

| Parameter | Default | What it does | Practical guidance |
|---|---:|---|---|
| `PROJECT_ROOT` | `Path(".")` | Base directory for the SPH table, SPH debris directory, and scenario directory. | In the current script, leave this as `Path(".")` and run from the project root unless you have tested non-default path construction. |
| `SCENARIO_NAME` | `"scenario_name"` | Name of the scenario folder and the prefix used for starting/checkpoint files. | Change this to run a different initial-condition folder. The folder and file prefix must match. |
| `SCENARIO_DIR` | `PROJECT_ROOT / SCENARIO_NAME` | Derived path to the scenario directory. | Usually do not edit directly; change `PROJECT_ROOT` or `SCENARIO_NAME`. |
| `START_TAG` | `"start"` | Selects the input file `<scenario>/<scenario>.<START_TAG>`. | Use `"start"` for an initial run or values like `"t1000000"` for restart checkpoints. |
| `SPH_TABLE_PATH` | `PROJECT_ROOT / "SPH.table"` | Path to the SPH remnant/outcome table. | Change only if the table is stored elsewhere. Must match `CONSTANTS.py` grid definitions. |
| `SPH_DEBRIS_DIR` | `PROJECT_ROOT / "SPHDebris_catalogue"` | Directory containing per-collision SPH debris files. | Change only if the debris catalogue is stored elsewhere. |

Restart detail: if `START_TAG = "t1000000"`, the script starts at `t = 1,000,000 yr` and then integrates for another `TOTAL_INTEGRATION_YEARS`. It does not stop at `TOTAL_INTEGRATION_YEARS` as an absolute final time.

### 7.2 Integration duration and checkpointing

| Parameter | Default | Unit/type | What it does | Practical guidance |
|---|---:|---|---|---|
| `TOTAL_INTEGRATION_YEARS` | `1.0e8` | years | Physical duration to integrate from the selected starting time. | Paper-scale default is 100 Myr. Use smaller values for smoke tests. |
| `TIMESTEP_REBOUND` | `1.0e-2 * 2.0 * np.pi` | REBOUND time units | Global `mercurius` timestep. This equals `0.01 yr` physically. | Smaller values improve time resolution but increase runtime. Changing this can change collision timing and energy error. |
| `N_CHECKPOINTS` | `1000` | integer | Number of evenly spaced checkpoint outputs. | Checkpoint spacing is `TOTAL_INTEGRATION_YEARS / N_CHECKPOINTS`. Larger values produce more files. |
| `REMOVAL_CHECK_FREQUENCY_STEPS` | `1000` | timesteps | Number of integrator steps between manual removal checks during long intervals. | Lower values remove invalid/ejected bodies sooner but add overhead. |

### 7.3 Massive bodies and particle classes

| Parameter | Default | Unit/type | What it does | Practical guidance |
|---|---:|---|---|---|
| `N_GAS_GIANTS` | `2` | integer count | Treats particles with indices `1` through `N_GAS_GIANTS` as gas giants. These bodies are excluded from manual removal and act as perfect-accretion sinks in collisions. | With Sun at index `0`, `2` means particles `1` and `2` should be `JUP` and `SAT`. Set to `0` if there are no gas giants. |

Important: `label_to_code()` currently recognizes only `JUP` and `SAT` as named gas giant labels. If `N_GAS_GIANTS > 2`, add labels for the additional giant planets.

### 7.4 Fragmentation and debris resolution

| Parameter | Default | Unit/type | What it does | Practical guidance |
|---|---:|---|---|---|
| `MIN_FRAGMENT_MASS_MEAR` | `5.5e-4` | Earth masses | Minimum resolved fragment mass. It is used both as a survivor cutoff and to compute `N_fr = floor(M_fr / MFM)`. | This is one of the most scientifically important knobs. Smaller values create more debris particles and are slower; larger values create fewer, more massive fragments and can promote faster reaccretion. |
| `ACTIVE_DEBRIS_MASS_MSUN` | `3.0e-7` | solar masses | Mass threshold used when deciding how many newly created debris particles remain active rather than test particles. | `3.0e-7 M_sun` is about `0.1 M_earth`. Changing this changes self-gravity among debris and runtime. |
| `DUST_MASS_RETENTION_FACTOR` | `0.5` | dimensionless factor | Fraction of the SPH-predicted debris reservoir retained as resolved N-body debris after survivor budgets are computed and before cluster-mass normalization. | Must be between `0.0` and `1.0`. Use `1.0` for strict resolved debris mass conservation. Use `0.5` to insert half of the SPH debris mass and treat the other half as unresolved dust carrying the same mean debris water fraction. The number of debris clusters is still computed from the pre-dust `M_fr`, so this factor changes resolved debris mass, not the nominal cluster count except when set to `0.0`. |
| `KMEANS_RANDOM_STATE` | `0` | integer or `None` | Random seed passed to scikit-learn `KMeans`. | Keep fixed for reproducibility. Changing it can alter debris cluster centroids and therefore subsequent dynamics. |
| `ACTIVE_DEBRIS_MASS_MSUN` | `3.0e-7` | solar masses | Threshold for classifying debris as active in the current activation policy. | Verify the `sim.N_active` policy before changing this in production runs; REBOUND active/test-particle behavior is index-based. |

### 7.5 Integrator and close-encounter settings

| Parameter | Default | Unit/type | What it does | Practical guidance |
|---|---:|---|---|---|
| `MERCURIUS_HILLFAC` | `5.0` | dimensionless | Sets `sim.ri_mercurius.hillfac`, controlling the hybrid switch scale for close encounters. | Larger values switch to close-encounter handling farther out; smaller values may be faster but risk less accurate encounters. |
| `IAS15_MIN_DT_FACTOR` | `1.0e-4` | dimensionless | Sets the minimum IAS15 timestep as this factor times `sim.dt`. | Decrease for stricter close-encounter integration; increase only after testing energy behavior. |
| `FALSE_COLLISION_RADIUS_FACTOR` | `2.0` | dimensionless | Rejects direct-geometry collision detections when separation is greater than this factor times `(R1 + R2)`. | Leave at default unless diagnosing false-positive or missed direct contacts. |
| `MIN_BACKTRACK_DISTANCE_HILL_RADII` | `3.0` | mutual Hill radii | Assigned to the legacy variable `min_btd`. | This variable is not used by the current cleaned script after assignment. Changing it has no effect unless additional backtracking logic is restored. |

### 7.6 Removal and sink criteria

Manual removal is applied to particles with indices greater than `N_GAS_GIANTS`, so the Sun and configured gas giants are not removed by this routine.

| Parameter | Default | Unit/type | What it does | Practical guidance |
|---|---:|---|---|---|
| `REMOVE_ECCENTRICITY_MIN` | `1.0` | dimensionless | Removes particles with `e >= REMOVE_ECCENTRICITY_MIN`. | Default removes unbound particles. |
| `REMOVE_A_MAX_AU` | `15.0` | AU | Removes particles with semimajor axis larger than this value. | Increasing retains distant bodies longer; decreasing makes the simulation more open. |
| `REMOVE_A_MIN_AU` | `0.1` | AU | Merges particles into the Sun if their semimajor axis is below this value. | Changing this affects inner-boundary mass loss. |
| `REMOVE_Q_MIN_AU` | `0.01` | AU | Merges particles into the Sun if perihelion `q = a(1-e)` is below this value. | Changing this affects near-Sun loss and water/mass budgets. |

### 7.7 Output and safety flags

| Parameter | Default | Unit/type | What it does | Practical guidance |
|---|---:|---|---|---|
| `SAVE_CHECKPOINTS` | `True` | boolean | Writes regular checkpoint files `<scenario>/<scenario>.t<time>`. | Turn off only for test runs where restart files are not needed. |
| `NEW_SIM_WARNING` | `False` | boolean | If `True`, prompts before deleting old `.dat` and `.snapshot` files in the scenario directory. | For clean production runs, manually archive or delete the full scenario output directory. This prompt does not remove every possible output type. |
| `SAVE_PROGRESS` | `True` | boolean | Appends progress and energy-error information to `<scenario>/progress.<scenario>`. | Keep enabled for long runs. |
| `SAVE_COLLISION_OUTCOME` | `True` | boolean | Appends compact collision summaries to the progress file. | Keep enabled unless you want a quieter progress log. |

### 7.8 Optional diagnostic plotting and printing

| Parameter | Default | Unit/type | What it does | Practical guidance |
|---|---:|---|---|---|
| `PRINT_INTERPOLATION_POINTS` | `True` | boolean | Prints SPH interpolation corner information for each queried collision. | Useful for debugging catalogue coverage. Set to `False` for quieter production runs. |
| `SHOW_DEBRIS_DIAGNOSTIC_PLOT` | `False` | boolean | Shows diagnostic plots of SPH debris samples and clustered debris. | Opens interactive Matplotlib figures and can stop unattended batch jobs. |
| `SHOW_COLLISION_GEOMETRY_PLOT` | `False` | boolean | Shows 3D post-collision survivor/debris geometry plots. | Use only for debugging individual collisions. |
| `SHOW_FRAME_MAPPING_STEPS` | `False` | boolean | Shows intermediate plots from the SPH-frame to simulation-frame mapping calculation. | Use only when validating coordinate transformations. |

---

## 8. Backwards-compatible aliases

Immediately after the configuration block, the script defines legacy aliases:

```python
t0 = START_TAG
Dt = TOTAL_INTEGRATION_YEARS
dt = TIMESTEP_REBOUND
Nsaves = N_CHECKPOINTS
scenario = str(SCENARIO_DIR)
NGG = N_GAS_GIANTS
MFM = MIN_FRAGMENT_MASS_MEAR
save_checkpoints = SAVE_CHECKPOINTS
new_sim_warning = NEW_SIM_WARNING
save_progess = SAVE_PROGRESS
save_collision_outcome = SAVE_COLLISION_OUTCOME
```

These aliases are used by the original procedural code below the configuration section. Edit the uppercase configuration variables, not the aliases, unless you are intentionally modifying legacy internals.

---

## 9. Outputs

All outputs are written inside the scenario directory.

### 9.1 Collision table

```text
<scenario>/coll.dat
```

Machine-readable table with one row per SHARD-resolved collision. It records:

- time,
- collision parameters `v0`, `alpha`, `M_tot`, `gamma`, `wft`, `wfp`,
- collision center-of-mass position and velocity,
- SPH-to-simulation-frame rotation angles,
- survivor count and survivor masses/water fractions,
- debris mass and water fraction,
- debris label range.

### 9.2 Events log

```text
<scenario>/events.dat
```

Human-readable record of collisions, removals, mergers into the Sun, and resulting particles.

### 9.3 Progress log

```text
<scenario>/progress.<scenario>
```

Reports:

- physical time in years,
- particle count,
- wall-clock timing,
- relative energy error,
- optional compact collision summaries.

### 9.4 Regular checkpoints

```text
<scenario>/<scenario>.t<rounded_time>
```

Written when `SAVE_CHECKPOINTS = True`. These files use orbital elements and can be used as restart inputs by setting `START_TAG` to the matching `t<rounded_time>` suffix.

### 9.5 Collision snapshots

```text
<scenario>/collision_<N>.snapshot
```

Written at collision times. These include orbital elements plus Cartesian position and velocity for every body.

### 9.6 Debris diagnostics

```text
<scenario>/collision_<N>.debris
```

Written before KMeans clustering. These files contain the interpolated weighted SPH debris samples used to construct the clustered N-body debris.

---

## 10. Restarting a run

1. Choose a checkpoint file, for example:

   ```text
   scenario_name/scenario_name.t1000000
   ```

2. Set:

   ```python
   START_TAG = "t1000000"
   ```

3. Set the desired additional integration duration:

   ```python
   TOTAL_INTEGRATION_YEARS = 9.9e7
   ```

   This example continues from 1 Myr for another 99 Myr, ending at 100 Myr.

4. Run:

   ```bash
   python SHARD_Rebound_cleaned.py
   ```

Before restarting, move or rename old output files if you want a clean log. The script appends to some files.

---

## 11. Practical run checklist

Before a production run, check the following:

- `CONSTANTS.py` is present and matches the SPH table grid.
- `SPH.table` is present at `SPH_TABLE_PATH`.
- `SPHDebris_catalogue/` contains the required `{id}_{code}.dat` files.
- The scenario folder exists and contains `<scenario>.<START_TAG>`.
- The starting file columns match the expected format.
- The first row is the Sun.
- Gas giants, if present, occupy particle indices `1..N_GAS_GIANTS`.
- `sim.N_active` from the input file is sensible.
- `MIN_FRAGMENT_MASS_MEAR`, `DUST_MASS_RETENTION_FACTOR`, and `ACTIVE_DEBRIS_MASS_MSUN` are intentionally chosen.
- Old outputs have been archived or removed.
- Diagnostic plotting flags are `False` for unattended runs.
- A short smoke test has completed before launching the full integration.

---

## 12. Scientific knobs that most affect results

The following parameters are especially important for scientific comparisons:

| Parameter | Why it matters |
|---|---|
| `MIN_FRAGMENT_MASS_MEAR` | Controls debris multiplicity and survivor cutoff. It can change reaccretion efficiency, damping, retained mass, and final particle spectrum. |
| `ACTIVE_DEBRIS_MASS_MSUN` | Controls which debris particles are gravitationally active. This can affect debris-debris interactions and runtime. |
| `DUST_MASS_RETENTION_FACTOR` | Controls how much of the SPH debris reservoir is retained as resolved N-body debris. Values below `1.0` remove mass and water into an unresolved dust component. |
| `TIMESTEP_REBOUND` | Affects collision timing, close-encounter resolution, and energy behavior. |
| `MERCURIUS_HILLFAC` | Affects the transition to close-encounter handling. |
| `REMOVE_A_MAX_AU`, `REMOVE_A_MIN_AU`, `REMOVE_Q_MIN_AU` | Define the open-system boundary and therefore mass/water loss. |
| `KMEANS_RANDOM_STATE` | Controls reproducibility of debris clustering. |

Do not change these for a published reproduction unless the change is part of a documented sensitivity test.

---

## 13. Known implementation cautions

### Active-particle indexing

The current code preserves the manuscript's activation prescription for newly created debris. REBOUND's `N_active` is index-based, so changes to `ACTIVE_DEBRIS_MASS_MSUN`, debris sorting, or particle ordering should be tested carefully.

### Path handling

The legacy script uses the scenario value as both directory and file prefix. Keeping `PROJECT_ROOT = Path(".")` and running from the project root is the safest current workflow.

### SPH catalogue coverage

Debris interpolation uses neighboring SPH grid corners without extrapolating outside the tabulated grid for debris phase-space sampling. Make sure the collision parameter range of the N-body run is covered by the catalogue.

### Missing debris files

The loader currently suppresses some debris-file loading errors for non-crashed SPH entries. If a required file is missing, failure may occur later during interpolation. For production use, validate catalogue completeness before long runs.

### Dust retention factor

`DUST_MASS_RETENTION_FACTOR` is applied to the target debris budget before clustered debris masses are normalized. Therefore `1.0` keeps the full SPH debris budget as resolved N-body fragments, while values below `1.0` remove the remaining mass and associated water into an unresolved dust component. The retained debris water fraction is unchanged; the resolved debris water mass scales with the retained mass.

The cluster count is computed from the pre-dust debris mass budget. This keeps the fragment-resolution rule tied to the SPH debris reservoir while allowing the retained resolved mass to be reduced separately.

---

## 14. Suggested smoke-test procedure

For a new machine or new dataset:

1. Use the default project layout.
2. Set:

   ```python
   TOTAL_INTEGRATION_YEARS = 100.0
   N_CHECKPOINTS = 2
   PRINT_INTERPOLATION_POINTS = False
   SHOW_DEBRIS_DIAGNOSTIC_PLOT = False
   SHOW_COLLISION_GEOMETRY_PLOT = False
   SHOW_FRAME_MAPPING_STEPS = False
   ```

3. Run:

   ```bash
   python -m py_compile SHARD_Rebound_cleaned.py
   python SHARD_Rebound_cleaned.py
   ```

4. Confirm that the scenario directory contains:

   ```text
   coll.dat
   events.dat
   progress.<scenario>
   <scenario>.t<time>
   ```

5. Inspect `progress.<scenario>` for obvious energy-error growth or crashes.
6. Restore paper-scale values before production.

---

## 15. Minimal paper-default configuration

For the cleaned script's current paper-style default, use:

```python
PROJECT_ROOT = Path(".")
SCENARIO_NAME = "scenario_name"
START_TAG = "start"

TOTAL_INTEGRATION_YEARS = 1.0e8
TIMESTEP_REBOUND = 1.0e-2 * 2.0 * np.pi
N_CHECKPOINTS = 1000

N_GAS_GIANTS = 2
MIN_FRAGMENT_MASS_MEAR = 5.5e-4
ACTIVE_DEBRIS_MASS_MSUN = 3.0e-7
DUST_MASS_RETENTION_FACTOR = 0.5
KMEANS_RANDOM_STATE = 0

MERCURIUS_HILLFAC = 5.0
IAS15_MIN_DT_FACTOR = 1.0e-4
REMOVAL_CHECK_FREQUENCY_STEPS = 1000

REMOVE_ECCENTRICITY_MIN = 1.0
REMOVE_A_MAX_AU = 15.0
REMOVE_A_MIN_AU = 0.1
REMOVE_Q_MIN_AU = 0.01

SAVE_CHECKPOINTS = True
SAVE_PROGRESS = True
SAVE_COLLISION_OUTCOME = True
```

For unattended production runs, consider setting:

```python
PRINT_INTERPOLATION_POINTS = False
```

so the terminal log does not become excessively large.
