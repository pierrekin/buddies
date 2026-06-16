# buddies — electrical

## Overview

| Gen  | Codename   | Folder       |
|------|------------|--------------|
| 1    | **Hello**  | `g1-hello/`  |
| 2    | **World**  | `g2-world/`  |
| 3    | **Native** | `g3-native/` |
| 4    | **Diver**  | `g4-diver/`  |

## Layout

```
electrical/
  lib/        shared part library
  blocks/     reusable schematic blocks
  ...
```

## Sharing rules

- **`lib/`** is the single source of truth for part choices. Each project's
  `sym-lib-table` / `fp-lib-table` point at it with `${KIPRJMOD}/../lib/...`.
- **Live-share** (shared hierarchical sheet) only blocks that are truly frozen.
- **Copy forward** (KiCad Design Block) everything else, so each generation owns
  its copy and a divergence in one generation never breaks another.

