# Review and verification

This record explains what was checked before the first GitHub release. It is a bounded project review, not a security certification or a claim of perfect game performance.

The project used the installed [Elephant & Goldfish Model (EGM)](https://github.com/google/elephant-goldfish-model) plugin, version 0.1.1. Independent reviewers started without conversation history and read the supplied source files, persona and standard. The main agent assessed their findings and made the fixes. This was direct reviewer dispatch, not a claim that every stage of the full EGM process or its convergence rule was completed.

## Findings and changes

| Finding | Changed files | Resolution and regression check |
|---|---|---|
| Worker actions could conflict with immediate manual takeover in the proposed design | `extension/controller.js`, `extension/background.js`, `docs/extension-design.md` | Decisions and the final enable flag live together in the page. Heartbeats cannot enable the agent. Real-input and stop-counter browser tests check this. |
| Runtime game compatibility needed an explicit gate | `dino_rl/config.py`, `scripts/export_policy.py`, `extension/policy.js`, `extension/controller.js` | Export dimensions/physics/schema; validate before control and during inference. Corrupt-state, unknown-obstacle and resize tests cover rejection. |
| Design alternatives were missing | `docs/extension-design.md` | Added a short trade-off table for protected Dino, native offline tab, clone and local service. |
| Declared Stable-Baselines3 minimum did not match an imported API | `pyproject.toml`, `uv.lock` | Require the tested 2.9.2-or-later API and keep the resolved environment in the lockfile. |
| Changing worker count lost a full replay buffer’s oldest-entry boundary | `dino_rl/parallel.py`, `tests/test_contract.py` | Copy wrapped replay chronologically; test flags, contents and the next eviction. |
| Same-run resume could mix different evaluation time limits | `dino_rl/training.py`, `tests/test_contract.py` | Reject a changed cap before Chrome opens. A new run directory keeps a separate selection history. |
| A failed worker could leave other games running | `dino_rl/parallel.py`, `tests/test_parallel.py` | Own bounded worker cleanup and child environment finalization; test step failure and partial startup. Preserve a startup exception if cleanup also fails. |
| One-decision diagnostics had no timing intervals for a percentile | `scripts/evaluate_transfer.py`, `tests/test_transfer.py` | Write JSON `null` for unavailable percentiles; test one and several decisions. |
| CLI mixed orchestration with training, storage and checks | `dino_rl/__main__.py`, `training.py`, `evaluation.py`, `checkpoints.py`, `checks.py` | Separate existing responsibilities without changing the algorithm; retain contract/resume regressions. |
| Release model has no full training state | `dino_rl/training.py`, `tests/test_contract.py`, `models/README.md` | Reject `--resume` on the inference release before launching Chrome; document `--initialize-from`. |
| Scrolling over the control panel escaped manual takeover | `extension/controller.js`, `tests/test_extension_browser.py` | Only actual button activation is exempt; wheel input pauses even over the panel. |
| Headless Chrome reports multiple tabs as focused/visible | `extension/background.js`, `tests/test_extension_browser.py` | Also pause on Chrome’s explicit active-tab/window events. Verify tab return does not resume. |
| CPU telemetry could be mistaken for exact accounting | `dino_rl/resources.py`, `README.md` | Describe sampled estimates; no extra process-accounting system was added. |
| Ctrl+C could wait for a full checkpoint evaluation batch | `dino_rl/evaluation.py`, `dino_rl/training.py`, `tests/test_contract.py` | Check the stop request between worker steps, discard partial evaluation results, and preserve the existing best model. |
| The initial worker resume could arrive after manual takeover | `extension/controller.js`, `extension/background.js`, `tests/test_extension_browser.py` | Start control synchronously with installation; delay the worker reply in a regression and verify manual input remains paused. |
| Exact compatibility settings were missing from the design text | `docs/extension-design.md` | Add the complete runner and dinosaur field/value tables, with exact-match semantics. |
| Resume could briefly re-enable a controller after its connection ended | `extension/controller.js`, `tests/test_extension_browser.py` | Only worker pulses refresh connection health; the disconnected page disables Resume and rejects attempts to re-enable it. |

## Check groups

| Check | Command / scope |
|---|---|
| Python style | `uv run ruff check .` and `uv run ruff format --check .` |
| Python contracts and regressions | `uv run python -m unittest discover -s tests` |
| JavaScript policy | `npm test` |
| Real Chrome extension lifecycle | `DINO_BROWSER_TESTS=1 uv run python -m unittest discover -s tests -p test_extension_browser.py -v` |
| Python ↔ JavaScript parity | Included in Python suite: 500 generated states, released weights, identical chosen actions |
| Dependency audit | `uv run pip-audit`; third-party packages checked, local unpublished package skipped |
| Packaging and publication | Frozen environment, build contents, staged files, private personal GitHub repository |

The small model ZIP and its deterministic JSON export are intentional release files. Replay, optimizer-resume state, machine paths, training logs, browser profiles, audit logs and review transcripts stay outside Git. The previous project guide is preserved locally under ignored `runs/archive/project-guide/`.

## Visual documentation

The README and guides use plain English, Mermaid workflows and tables. The visible-test chart comes from `models/validation.json`; the capped first game is marked explicitly.

The optional explorable architecture diagram was created with installed **Archify 2.17**. Its JSON source and self-contained HTML are committed. Archify’s delivery check passed all nine showcase checks with zero errors/warnings. Automated browser checks passed containment at 1440×900, 1600×1000, 1920×1080 and 2048×1320. Rendered light and dark screenshots were inspected separately for readable nodes, labels and connections. Raw receipts and screenshots stay in local review artifacts.

## Practical limits

- Native Chrome behavior was tested on the version recorded in the model card, on macOS. Other platforms and later Chrome versions are not claimed as verified.
- The extension’s numerical behavior and controls are checked. A long extension score benchmark remains outstanding.
- Only three separate visible validation games completed. That sample does not establish a reliable maximum score or perfect bird handling.
- Browser policies may prohibit the required extension/debugger access. No bypass is provided.
- Browser scheduling and native obstacle randomness prevent exact replay of training trajectories.

## Final local results

- Clean temporary Python environment: 23 tests passed; native Chrome tests were run separately.
- Native Chrome extension checks: five tests passed, including the opening/manual-input race, connection expiry and disabled Resume after debugger cancellation.
- JavaScript policy suite: four tests passed.
- Python style and formatting: passed.
- Native movement/interface check: passed on Chrome 157.0.8103.78.
- Source and wheel builds: passed; generated experiment directories were absent from the archives.
- Dependency audit: no known vulnerabilities among audited third-party packages; the unpublished local package was skipped.
- Publication scan: no machine paths or credential-pattern matches in the intended release files, including the model ZIP.

The two final fresh-context reviewers completed their source/document audits. Their material findings were the opening resume race, missing exact compatibility values, and cancellation behavior during evaluation. Each was fixed as described above. Test execution and packaging verification were performed separately by the main agent. No known unresolved Major or Blocker from these bounded reviews remains; this does not claim the full EGM convergence protocol.

GitHub CI runs separately after publication; local success does not assert a remote CI result.

<details>
<summary>Architecture diagram receipt</summary>

| Field | Result |
|---|---|
| Type | architecture |
| Source | `docs/architecture.json` |
| Output | `docs/architecture.html` |
| Validation | 9/9 showcase, zero errors, zero warnings |
| Automated browser evidence | passed |
| Visual review | passed, light and dark endpoint screenshots |
| Correction rounds | 1 |
| Source SHA-256 | `0341280864d8f310170b45b816e2c01a802e8e996beb8afb84f6e4980aaccd26` |
| HTML SHA-256 | `d311f2e684c49805487e1d08f56141d18cf3f94299a4d9e78632bdbc84ad3ce0` |

</details>
