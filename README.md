# Research code and data

Code, structural models, and processed data for multivariate seismic demand
analysis.

## Contents

- `src/`: model evaluation, fitting utilities, and parameter sets.
- `models/`: OpenSeesPy and Tcl structural model definitions.
- `data/`: processed responses and a compact ground-motion record index.
- `analysis/`: fitting entry point for the processed response data.
- `synthetic/`: Exp1-Exp5 scripts, a fixed sample, and selected result tables.

## Setup

Install the statistical dependencies with:

```text
python -m pip install -r requirements.txt
```

For the structural model, also install:

```text
python -m pip install -r requirements-opensees.txt
```

The scripts write newly generated files under local `outputs/` directories.

Fit the model to the supplied MSA responses with:

```text
python analysis/fit_model.py
```

The fitting command uses the training split by default.

## Data

The response tables contain processed engineering-demand measures, record
identifiers, data splits, intensity measures, and scale factors. Raw
ground-motion waveforms are not redistributed and remain subject to their
source provider's terms.

## License

The code is released under the MIT License.
