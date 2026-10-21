# Third-party notices

The project MIT license applies to original project code. The following included
components retain their upstream copyright notices and licenses.

## AIDE

- Source: <https://github.com/WecoAI/aideml>
- Revision: `60b3978ddf65b71f86eb7c64506965048a1398cf`
- Copyright (c) 2024 Weco AI Ltd.
- License: MIT; complete text in
  `src/pde_agent_verifier_solver/plugins/ns/baselines/_vendor/AIDE_LICENSE`.

The release includes selected upstream source files under
`src/pde_agent_verifier_solver/plugins/ns/baselines/_vendor/aide_upstream/`
and a derived core in `src/pde_agent_verifier_solver/plugins/ns/baselines/_vendor/aide_core.py`.
`AIDE_PROVENANCE.json` in the vendor directory records source hashes and the
extraction and adaptation changes. The project supplies NS generation and
evaluation callbacks;
this adaptation is not an end-to-end reproduction of the upstream experiments.

## ERA

- Source: <https://github.com/google-research/era>
- Revision: `eb56676ce0a9b22b0c357c218de09ac61e94fc4b`
- Copyright 2026 Google LLC.
- License: Apache License, Version 2.0; complete text in
  `src/pde_agent_verifier_solver/plugins/ns/baselines/_vendor/ERA_LICENSE`.

The file `implementation/futs.py` is included without modification as
`src/pde_agent_verifier_solver/plugins/ns/baselines/_vendor/era_futs.py`.
Its copyright and license header are retained. `ERA_PROVENANCE.json` records
the source revision and hash. The surrounding NS callback integration is
project code and is not part of the upstream file.

## CodePDE attribution

The independent control-policy implementation in
`src/pde_agent_verifier_solver/plugins/ns/baselines/codepde.py` refers to
[CodePDE](https://github.com/LithiumDA/CodePDE), revision
`26938c5832ddd0a957426dff6f702b54649c14af`.

The release includes provenance metadata, not CodePDE source files.
`CODEPDE_PROVENANCE.json` records the inspected sources and the independent
reimplementation boundary. The recorded upstream audit found no license file;
no redistribution permission or upstream license is inferred.

## External packages and artifacts

Packages installed separately retain their own licenses. The framework also
supports external physical implementations, datasets and model checkpoints;
those are not relicensed by the project MIT license. Follow the source-specific
terms when acquiring or redistributing them.
