# tensor_regression
A small wrapper to simplify using [pytest_regressions](https://github.com/ESSS/pytest-regressions) with Tensors.

This adds the following to [pytest_regressions](https://github.com/ESSS/pytest-regressions):
- Simple Tensor statistics (min, max, mean, std, shape, dtype, device, hash, etc.) are generated and saved in a .yaml file.
  - The simple statistics are used as a pre-check before comparing the full tensors.
  - These yaml files can be saved with git without having to worry about accidentally saving huge files.
- Full tensors are moved to CPU and saved in a `.npy` file (same as ndarrays_regression), and these .npy files are gitignored.
- Adds a `--gen-missing` argument (default True) which will generate any missing regression files without raising error, as opposed to pytest-regression's `--regen-all` which regenerates all regression files.

## Flow

The following diagram illustrates the different cases that occur when calling `tensor_regression.check()`:

```mermaid
flowchart TD
    A["tensor_regression.check(data)"] --> B["Flatten data dict\nDetermine .yaml & .npz file paths"]

    B --> C{"--skip-if-files-missing\nand any file missing?"}
    C -->|Yes| SKIP(["⏭ Skip test"])
    C -->|No| D{"--regen-all?"}
    D -->|Yes| E["Delete .yaml and .npz\n(if they exist)"]
    E --> F
    D -->|No| F{".npz file exists?"}

    F -->|Yes| G{".yaml file exists?"}
    G -->|No| H["Recreate .yaml\nfrom current data stats"]
    H --> I["Compare full tensors\nvs saved .npz"]
    G -->|Yes| I
    I -->|Mismatch| FAIL1(["❌ FAIL\nTensor values changed"])
    I -->|Match| J["Compare stats\nvs saved .yaml"]
    J -->|Mismatch| FAIL2(["❌ FAIL\nStats changed"])
    J -->|Match| PASS1(["✅ PASS"])

    F -->|No| K{".yaml file exists?"}
    K -->|Yes| L["Recreate .npz\nfrom current tensors"]
    L --> M["Compare stats\nvs saved .yaml"]
    M -->|Mismatch| FAIL3(["❌ FAIL\nStats mismatch"])
    M -->|Match| PASS2(["✅ PASS\n(.npz regenerated)"])

    K -->|No| N["Create .yaml and .npz\nfrom current data"]
    N -->|"--gen-missing=True (default)"| PASS3(["✅ PASS\n(files created)"])
    N -->|"--gen-missing=False"| FAIL4(["❌ FAIL\nFiles missing"])
```
