"""Data utilities for ARC-AGI & ARC-AGI-2.

Provides:
- Grid serialization and robust deserialization (`grid_to_text`, `text_to_grid`)
- Color bijective permutations (`random_color_map`, `apply_color_map`)
- Dihedral D8 group spatial transformations (`apply_d8_transform`)
- Task augmentation (`augment_task`, `generate_task_augmentations`)
- Prompt generation and conversation formatting (`task_to_prompt`, `task_to_chat_messages`)
"""

import json
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ARC Standard Color Palette (0-9)
COLOR_NAMES = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "gray",
    6: "magenta",
    7: "orange",
    8: "azure",
    9: "maroon",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert AI solving ARC-AGI (Abstraction and Reasoning Corpus) puzzles. "
    "Analyze the demonstrations to identify the abstract transformation rule, "
    "then output the exact transformed grid for the test input. "
    "Output ONLY the final grid as space-separated integers per row without extra text."
)


# ============================================================================
# 1. Grid Validation & Equality
# ============================================================================

def is_valid_grid(grid: Any) -> bool:
    """Validate that the input is a rectangular 2D list of integers 0-9."""
    if not isinstance(grid, (list, tuple)) or len(grid) == 0:
        return False
    if not isinstance(grid[0], (list, tuple)) or len(grid[0]) == 0:
        return False
    
    num_cols = len(grid[0])
    num_rows = len(grid)
    
    # ARC grids are bounded between 1x1 and 30x30
    if num_rows > 30 or num_cols > 30:
        return False

    for row in grid:
        if not isinstance(row, (list, tuple)) or len(row) != num_cols:
            return False
        for val in row:
            if not isinstance(val, int) or val < 0 or val > 9:
                return False
    return True


def grids_equal(g1: Optional[Sequence[Sequence[int]]], g2: Optional[Sequence[Sequence[int]]]) -> bool:
    """Check exact match equality between two 2D grids (both dimensions and cell values)."""
    if g1 is None or g2 is None:
        return False
    if len(g1) != len(g2):
        return False
    if len(g1) == 0:
        return len(g2) == 0
    if len(g1[0]) != len(g2[0]):
        return False
    
    for r in range(len(g1)):
        for c in range(len(g1[0])):
            if g1[r][c] != g2[r][c]:
                return False
    return True


# ============================================================================
# 2. Grid Serialization & Deserialization
# ============================================================================

def grid_to_text(grid: Sequence[Sequence[int]], format_style: str = "compact") -> str:
    """Convert a 2D integer grid to a formatted string.
    
    Supported format_styles:
    - 'compact': Space-separated integers per line, e.g.:
        0 1 2
        3 4 5
    - 'brackets': Standard Python/JSON list of lists representation, e.g.:
        [[0, 1, 2], [3, 4, 5]]
    - 'delimited': Pipe-delimited matrix format, e.g.:
        | 0 1 2 |
        | 3 4 5 |
    """
    if not grid or len(grid) == 0:
        return ""

    if format_style == "brackets":
        return json.dumps([[int(c) for c in row] for row in grid])
    elif format_style == "delimited":
        lines = []
        for row in grid:
            lines.append("| " + " ".join(str(int(c)) for c in row) + " |")
        return "\n".join(lines)
    else:  # compact / default
        return "\n".join(" ".join(str(int(c)) for c in row) for row in grid)


# A grid row, once framing and any "Row 7:" label are stripped: nothing but
# single digits and separators. Requiring the WHOLE line to match is what keeps
# prose out - "I count 4 objects" must not become the 1x1 grid [[4]].
_ROW_LABEL = re.compile(r"^(?:row\s*)?\d{1,2}\s*[:.|]\s*", flags=re.IGNORECASE)
_FRAMING = re.compile(r"^[\s|\[\]'\"]+|[\s|\[\]'\",]+$")
_DIGIT_ROW = re.compile(r"^[0-9](?:[\s,]+[0-9])*$")


def _row_cells(line: str) -> Optional[List[int]]:
    """Cells of `line` if it is a grid row, else None."""
    stripped = _FRAMING.sub("", line.strip())
    stripped = _ROW_LABEL.sub("", stripped).strip()
    stripped = _FRAMING.sub("", stripped)
    if not stripped or not _DIGIT_ROW.match(stripped):
        return None
    return [int(d) for d in re.findall(r"[0-9]", stripped)]


def _scan_grids(text: str) -> List[List[List[int]]]:
    """Every maximal run of consecutive grid rows, in order of appearance."""
    found: List[List[List[int]]] = []
    current: List[List[int]] = []
    for line in text.splitlines():
        cells = _row_cells(line)
        if cells is None:
            if is_valid_grid(current):
                found.append(current)
            current = []
        else:
            current.append(cells)
    if is_valid_grid(current):
        found.append(current)
    return found


def text_to_grid(text: str) -> Optional[List[List[int]]]:
    """Robustly parse an ARC grid from model output text.

    Handles code blocks, bracketed JSON, and space/comma separated digit lines
    with reasoning text around them.

    Two rules earn their keep against real model output:

    * A line contributes cells only if the WHOLE line is digits and separators
      (after stripping framing and any "Row 7:" label). Scanning a line for
      loose digits instead makes "I count 4 objects" parse as the 1x1 grid
      [[4]] and return it before the real answer is ever reached.
    * When several grids are present, the LAST one wins. A model that reasons
      before answering emits its working first and its answer last, and a model
      that echoes the input emits the input first.
    """
    if not text or not isinstance(text, str):
        return None

    cleaned_text = text.strip()

    # Fenced blocks are the strongest signal, and the last fence is the answer.
    code_blocks = re.findall(r"```(?:json|python)?\s*(.*?)\s*```", cleaned_text, flags=re.DOTALL)
    candidate_texts = list(reversed(code_blocks)) + [cleaned_text]

    for candidate in candidate_texts:
        candidate = candidate.strip()
        if not candidate:
            continue

        # Explicit JSON is unambiguous; take the last well-formed one.
        json_matches = re.findall(r"\[\s*\[.*?\]\s*\]", candidate, flags=re.DOTALL)
        for jm in reversed(json_matches):
            try:
                parsed = json.loads(jm)
            except Exception:
                continue
            if is_valid_grid(parsed):
                return [[int(c) for c in row] for row in parsed]

        grids = _scan_grids(candidate)
        if grids:
            return grids[-1]

    return None


# ============================================================================
# 3. Spatial Dihedral (D8) Transformations
# ============================================================================

def apply_d8_transform(grid: Sequence[Sequence[int]], op: int) -> List[List[int]]:
    """Apply one of the 8 Dihedral group symmetries to a 2D grid.
    
    op values:
    0: Identity
    1: Rotate 90 deg clockwise
    2: Rotate 180 deg
    3: Rotate 270 deg clockwise
    4: Flip Horizontal (reflect across vertical axis)
    5: Flip Vertical (reflect across horizontal axis)
    6: Transpose (reflect across main diagonal)
    7: Anti-Transpose (reflect across secondary diagonal)
    """
    g = [list(row) for row in grid]
    h = len(g)
    w = len(g[0]) if h > 0 else 0

    if op == 0:  # Identity
        return g
    elif op == 1:  # Rot 90 CW
        return [[g[h - 1 - r][c] for r in range(h)] for c in range(w)]
    elif op == 2:  # Rot 180
        return [[g[h - 1 - r][w - 1 - c] for c in range(w)] for r in range(h)]
    elif op == 3:  # Rot 270 CW
        return [[g[r][w - 1 - c] for r in range(h)] for c in range(w)]
    elif op == 4:  # Flip Horizontal (left-right)
        return [[g[r][w - 1 - c] for c in range(w)] for r in range(h)]
    elif op == 5:  # Flip Vertical (up-down)
        return [[g[h - 1 - r][c] for c in range(w)] for r in range(h)]
    elif op == 6:  # Transpose (main diagonal)
        return [[g[r][c] for r in range(h)] for c in range(w)]
    elif op == 7:  # Anti-Transpose (secondary diagonal)
        return [[g[h - 1 - r][w - 1 - c] for r in range(h)] for c in range(w)]
    else:
        raise ValueError(f"Invalid D8 operation index {op}. Must be between 0 and 7.")


# ============================================================================
# 4. Color Permutations
# ============================================================================

def random_color_map(
    preserve_background: bool = True,
    rng: Optional[random.Random] = None
) -> Dict[int, int]:
    """Generate a bijective color permutation mapping for ARC colors (0-9).
    
    Args:
        preserve_background: If True, color 0 (black/background) maps to 0.
                             Colors 1-9 are permuted randomly among themselves.
                             If False, all colors 0-9 are permuted.
        rng: Optional random.Random instance for reproducible sampling.
    """
    r = rng if rng is not None else random.Random()
    if preserve_background:
        non_bg_colors = list(range(1, 10))
        shuffled = list(non_bg_colors)
        r.shuffle(shuffled)
        mapping = {0: 0}
        for orig, new in zip(non_bg_colors, shuffled):
            mapping[orig] = new
        return mapping
    else:
        all_colors = list(range(10))
        shuffled = list(all_colors)
        r.shuffle(shuffled)
        return {orig: new for orig, new in zip(all_colors, shuffled)}


def apply_color_map(
    grid: Sequence[Sequence[int]],
    color_map: Dict[int, int]
) -> List[List[int]]:
    """Apply a color mapping dictionary to all cells in a 2D grid."""
    return [[color_map.get(val, val) for val in row] for row in grid]


# ============================================================================
# 5. Task Augmentations
# ============================================================================

def augment_task(
    task: Dict[str, Any],
    d8_op: int = 0,
    color_map: Optional[Dict[int, int]] = None,
    shuffle_demonstrations: bool = False,
    rng: Optional[random.Random] = None
) -> Dict[str, Any]:
    """Apply consistent spatial and color transformations to an entire ARC task.
    
    In ARC, all input and output grids within both 'train' and 'test' pairs must
    share the exact same spatial transform and color permutation to maintain rule consistency.
    """
    r = rng if rng is not None else random.Random()
    
    def transform_pair(pair: Dict[str, Any]) -> Dict[str, Any]:
        res = {}
        if "input" in pair:
            inp = apply_d8_transform(pair["input"], d8_op)
            if color_map is not None:
                inp = apply_color_map(inp, color_map)
            res["input"] = inp
        if "output" in pair:
            out = apply_d8_transform(pair["output"], d8_op)
            if color_map is not None:
                out = apply_color_map(out, color_map)
            res["output"] = out
        return res

    augmented_train = [transform_pair(p) for p in task.get("train", [])]
    augmented_test = [transform_pair(p) for p in task.get("test", [])]

    if shuffle_demonstrations and len(augmented_train) > 1:
        r.shuffle(augmented_train)

    return {
        "train": augmented_train,
        "test": augmented_test
    }


def generate_task_augmentations(
    task: Dict[str, Any],
    num_augmentations: int = 8,
    permute_colors: bool = True,
    preserve_background: bool = True,
    apply_symmetries: bool = True,
    shuffle_demonstrations: bool = True,
    rng: Optional[random.Random] = None
) -> List[Dict[str, Any]]:
    """Generate a diverse set of augmented variants of a given ARC task."""
    r = rng if rng is not None else random.Random()
    augmented_tasks: List[Dict[str, Any]] = [task]  # Always include original

    for _ in range(num_augmentations - 1):
        d8_op = r.randint(0, 7) if apply_symmetries else 0
        cmap = random_color_map(preserve_background=preserve_background, rng=r) if permute_colors else None
        aug = augment_task(
            task,
            d8_op=d8_op,
            color_map=cmap,
            shuffle_demonstrations=shuffle_demonstrations,
            rng=r
        )
        augmented_tasks.append(aug)

    return augmented_tasks


# ============================================================================
# 6. Task to Prompt & Chat Formatting
# ============================================================================

def task_to_prompt(
    task: Dict[str, Any],
    test_idx: int = 0,
    include_test_output: bool = False,
    grid_format: str = "compact"
) -> Tuple[str, Optional[str]]:
    """Convert an ARC task into a structured input prompt and target completion string.
    
    Returns:
        (user_prompt, target_output_str or None)
    """
    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])
    if test_idx >= len(test_pairs):
        raise IndexError(f"test_idx {test_idx} is out of bounds for test set of length {len(test_pairs)}")

    target_test_pair = test_pairs[test_idx]

    parts: List[str] = []
    
    # 1. Demonstrations
    for idx, pair in enumerate(train_pairs, 1):
        parts.append(f"Demonstration {idx}:")
        parts.append("Input:")
        parts.append(grid_to_text(pair["input"], format_style=grid_format))
        parts.append("Output:")
        parts.append(grid_to_text(pair["output"], format_style=grid_format))
        parts.append("")

    # 2. Test input
    parts.append("Test Problem:")
    parts.append("Input:")
    parts.append(grid_to_text(target_test_pair["input"], format_style=grid_format))
    parts.append("Output:")

    user_prompt = "\n".join(parts)
    
    target_output: Optional[str] = None
    if include_test_output and "output" in target_test_pair:
        target_output = grid_to_text(target_test_pair["output"], format_style=grid_format)

    return user_prompt, target_output


def task_to_chat_messages(
    task: Dict[str, Any],
    test_idx: int = 0,
    include_test_output: bool = True,
    system_prompt: Optional[str] = None,
    grid_format: str = "compact"
) -> List[Dict[str, str]]:
    """Convert an ARC task into standard ChatML / OpenAI messages format.
    
    Format:
    [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."} # Only if include_test_output=True
    ]
    """
    sys_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
    user_prompt, target_output = task_to_prompt(
        task,
        test_idx=test_idx,
        include_test_output=include_test_output,
        grid_format=grid_format
    )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]

    if include_test_output and target_output is not None:
        messages.append({"role": "assistant", "content": target_output})

    return messages
