# Release review

This records the **November 19, 2026 / 0.2.0** release audit. It used [Elephant & Goldfish Model (EGM)](https://github.com/google/elephant-goldfish-model) 0.1.1 fresh-context reviewers and the installed Ponytail simplicity workflow. The main agent applied fixes and ran the checks. Raw reports stay in ignored local review artifacts.

This was a bounded release audit using EGM personas, not the full five-stage design process or a claim of formal convergence.

## Findings and fixes

| Finding | Resolution | Evidence |
|---|---|---|
| Ctrl+C callback counted a step before SB3 stored its transition | Stop after the ordinary collection batch commits; preserve saved DQN call counter | Actual SB3 interrupt/save/resume regression |
| Exploration action generator was missing from saved random states | Save/restore action-space RNG; warn explicitly for historical checkpoints | Restored action-sequence regression |
| Resource sampling failure could leave opened games behind | Create the monitor inside the environment’s cleanup scope | Constructor-failure regression for training and evaluation |
| Heartbeat expiry could be cleared by later input or a late pulse | Latch expiry before refresh; disable Resume until a new session | Native late-heartbeat and manual-input test |
| Automatic startup could take control after human input during loading | Install paused; require the page’s Start AI button | Input before/during installation tests |
| Navigation could redirect a delayed installation or publish stale ready status | Bind installation to document identity and check session after the reply | Navigation tests before/after installation |
| Secondary clicks on the control button escaped manual takeover | Only primary button activation is exempt | Native right-click takeover check |
| Export could accept changed action order or interval | Reject incompatible contracts before loading the model | Export rejection and numerical parity tests |
| Extension update instructions omitted downloading new files | Update the loaded folder before Reload, then confirm version and model step | Documentation reviewer re-read |

The Python reviewer inspected checkpoint, replay, timeout, cleanup and SB3 collection behavior. The extension reviewer inspected the page controller, worker, popup and lifecycle tests. Both re-read the fixes. A third fresh-context reviewer checked plain language, current results and consistency with source, then confirmed its update-instruction finding was fixed. No known unresolved finding remains from these bounded audits. Reviews are static; test results below come from separate execution.

## Simplicity and repository cleanup

- Removed the unused controlled-clock evaluation mode and CPU pacing branch. Training and evaluation now use native game time only.
- Removed the large optional HTML diagram viewer and its source/license notice. Mermaid workflows and a small results chart remain.
- Consolidated training instructions into one guide and replaced old model results with current measurements.
- Kept both model formats: Python needs the inference ZIP; the extension needs JSON weights.
- Excluded replay, full training checkpoints, browser profiles, logs, raw reviews, caches and build archives from Git. Unique local experiment history is retained.
- Kept the existing DQN architecture and learning settings. No avoidance heuristic, game clone, inference service or UI framework was added.

## Verification

| Check | Scope |
|---|---|
| Python tests | State/action contracts, fresh initialization, full-state resume, replay order, interrupts and worker cleanup |
| Export parity | 500 generated states; float32 observations match, Q-values agree within tolerance, chosen actions match |
| JavaScript tests | Model contract, release pointer/hash consistency, observation conversion and Q calculation |
| Native Chrome tests | Explicit start, manual/focus takeover, disconnect, navigation, expiry, dimensions and other tabs staying online |
| Native game check | Default physics, movement, collisions and decision timing |
| Model selection | Three policies × eight frozen-policy native games on Chrome 157.0.8103.79 |
| Extension benchmark | Three new games; two five-minute caps, one collision at 286.4 seconds |
| Release hygiene | Frozen dependency install, lint/format, dependency audit, build/archive contents and publication-file inspection |

Final local results: **31 Python checks, 8 native Chrome lifecycle checks and 4 JavaScript checks passed**. The clean non-editable environment ran 39 Python-discovered tests: 31 passed and the eight opt-in Chrome checks were run separately. Native movement, lint and formatting also passed; the source/wheel built successfully, and the publication scan found no private paths or credential-pattern matches, including inside the inference ZIP.

The dependency audit reported no known vulnerabilities in audited third-party packages; the unpublished local project was skipped. CI runs Python, JavaScript and formatting checks. Local Chrome tests use temporary profiles, not the user’s browsing profile.

## Performance limits

The selected **1,250,400-step** policy led the fresh comparison with **135.1 seconds median survival**. The independent three-game extension benchmark was stronger, but its different runtime/workload and small sample do not establish a universal score. Birds still caused failures in the selection batch. The five-million-step study stopped at **4,198,264 saved steps**; it did not finish its target.

Chrome **157.0.8103.79 on macOS** is the verified release environment. Native obstacles and browser scheduling vary. No claim is made of perfect play, unlimited survival, a completed five-million-step run or untested Chrome compatibility. See the [model card](../models/README.md) for the measurements and [extension design](extension-design.md) for control boundaries.
