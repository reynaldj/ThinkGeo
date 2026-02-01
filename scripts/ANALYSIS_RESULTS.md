# ThinkGeo Invalid Tool Call Analysis Results

## Executive Summary

Analysis of prediction files from different model configurations reveals distinct error patterns and opportunities for validator improvement.

**Dataset**: ThinkGeo benchmark with 1,773 ground truth steps across 486 tasks

---

## Qwen1.5-7B BaseStepByStep (mode='every_with_gt')

### Overall Statistics
- **Total Steps**: 1,921
- **Explicit Errors**: 311 (16.2%)
- **Invalid Without Error**: 1,236 (64.3%)
  - Missing Tool: 1,236 (100% of silent failures)
  - Tool Mismatch: 0
  - Arg Mismatch: 0
  - Sequence Violation: 0

### Key Finding: Direct Answer Problem
**The agent provides direct answers instead of making tool calls**

Examples:
- Gold expects `TextToBbox` → Pred answers "The distance between... is approximately X meters"
- Gold expects `TextToBbox` → Pred answers "The flooded houses... have been accurately identified"
- Gold expects `CountGivenObject` → Pred answers "There are buildings directly attached..."

### Missing Tool Breakdown (Top Tools)
| Expected Tool | Count | % of Total Missing |
|---------------|-------|-------------------|
| Calculator | 334 | 27.0% |
| TextToBbox | 260 | 21.0% |
| RegionAttributeDescription | 235 | 19.0% |
| CountGivenObject | 116 | 9.4% |
| ChangeDetection | 76 | 6.1% |
| Solver | 72 | 5.8% |
| DrawBox | 45 | 3.6% |
| SegmentObjectPixels | 30 | 2.4% |

**Insight**: Calculator (27%) and TextToBbox (21%) together account for 48% of missing tools, indicating the agent struggles most with computational and localization tasks.

---

## Qwen2.5-7B IsInstanceFixed (mode='every_with_gt')

### Overall Statistics
- **Total Steps**: 1,921
- **Explicit Errors**: 212 (11.0%)
- **Invalid Without Error**: 1,088 (56.6%)
  - Missing Tool: 566 (52.0%)
  - Tool Mismatch: 137 (12.6%)
  - Arg Mismatch: 85 (7.8%)
  - Sequence Violation: 300 (27.6%)

### Improvement vs Qwen1.5
- **7.7% reduction** in silent failures (64.3% → 56.6%)
- **Diversification of errors**: No longer just missing tools, now shows tool confusion and arg issues
- **Sequence violations appear**: 300 cases (27.6% of silent failures)

---

## Tool Mismatch Analysis (Qwen2.5)

### Top Confusions (Gold → Predicted)
| Expected | Predicted | Count | Pattern |
|----------|-----------|-------|---------|
| **Solver** → **Calculator** | 24 | Treats equation solving as simple calculation |
| **Calculator** → **CountGivenObject** | 21 | Confuses arithmetic with object counting |
| **TextToBbox** → **Calculator** | 11 | Misinterprets localization as computation |
| **Calculator** → **RegionAttributeDescription** | 11 | Confuses calculation with description task |
| **RegionAttributeDescription** → **DrawBox** | 10 | Visualization confusion |
| **Calculator** → **TextToBbox** | 6 | Thinks math requires localization |
| **TextToBbox** → **CountGivenObject** | 6 | Confuses detection with counting |
| **RegionAttributeDescription** → **Calculator** | 6 | Numeric attributes trigger wrong tool |

### Critical Patterns

#### 1. **Solver ↔ Calculator Confusion** (24 cases)
- Gold: `Solver` (for equations like "2x + 3 = 7")
- Pred: `Calculator` (tries to evaluate rather than solve)
- **Validator Rule**: Detect equation-solving keywords (solve, x=, find x) vs arithmetic

#### 2. **Calculator → CountGivenObject** (21 cases)
- Gold: `Calculator` (e.g., "count how many divided by total")
- Pred: `CountGivenObject` (focuses on "count" keyword, ignores math)
- **Validator Rule**: Presence of operators (+, -, *, /) should override counting keywords

#### 3. **TextToBbox → Calculator** (11 cases)
- Gold: `TextToBbox` (localize object)
- Pred: `Calculator` (sees numbers/measurements in description)
- **Validator Rule**: Spatial keywords (locate, find, where, position) → TextToBbox

#### 4. **RegionAttributeDescription → DrawBox** (10 cases)
- Gold: Describe region attributes
- Pred: Draw bbox around region
- **Validator Rule**: Description task != visualization task

---

## Argument Mismatch Analysis (Qwen2.5)

### Calculator: expression_not_math (78 cases, 91.8% of arg errors)

**Problem**: Agent passes non-mathematical expressions to Calculator

Examples of invalid expressions:
- Natural language: "the number of buildings in region A"
- Incomplete: "distance between" (missing operands)
- Invalid syntax: "count(buildings) + 5" (mixing counting with math)
- Variables without values: "x + y" (when x, y undefined)

**Validator Rule**:
```python
def validate_calculator_expr(expr: str) -> bool:
    # Must contain only: numbers, operators (+,-,*,/,(,),.), spaces
    # Must have balanced parentheses
    # Must be parseable as arithmetic expression
    return bool(re.fullmatch(r"[0-9+\-*/(). %]+", expr.strip()))
```

### TextToBbox: cardinality_mismatch (7 cases, 8.2% of arg errors)

**Problem**: Agent requests multiple bboxes when gold expects single, or vice versa

Examples:
- Gold: `{"top1": true}` (single bbox) → Pred: `{"top1": false}` (all bboxes)
- Gold: `{"top1": false}` (all matches) → Pred: `{"top1": true}` (only top match)

**Validator Rule**: Check query intent
- "the X" / "first X" → `top1: true`
- "all X" / "every X" / "X in the region" → `top1: false`

---

## Sequence Violation Analysis (Qwen2.5)

**300 cases (27.6% of silent failures)**

Despite `mode='every_with_gt'` providing ground truth at each step, sequence violations occur when:
1. Agent **ignores provided context** (bboxes from previous GT steps)
2. Agent **hallucinates dependencies** that don't exist in the trace
3. Agent **parsing fails** on GT output format

### Tools Most Affected
- DrawBox (requires bbox input)
- CountGivenObject (often needs bbox to define region)
- RegionAttributeDescription (needs bbox for region)

**Insight**: This suggests **context utilization failure** - the agent doesn't properly extract/use outputs from previous GT steps, pointing to prompt engineering issues or context window limitations.

---

## Validator Design Recommendations

### Stage 1: Constraint Rules (Fast Rejection)

#### 1. **Tool Selection Rules** (Prevent Confusion)
```python
# Rule 1: Equation solving vs arithmetic
if has_variable(query) and has_equals(query):
    tool = "Solver"  # not Calculator

# Rule 2: Math operators override counting
if has_operators(query, ['+', '-', '*', '/', '(', ')']):
    tool = "Calculator"  # not CountGivenObject

# Rule 3: Spatial keywords → localization
if has_spatial_keywords(query, ['locate', 'where', 'position', 'find']):
    tool = "TextToBbox"  # not Calculator

# Rule 4: Description != Visualization
if intent_is_description(query):
    tool in {"RegionAttributeDescription", "ImageDescription"}
    # not DrawBox, not Plot
```

#### 2. **Argument Validation Rules**
```python
# Calculator: strict math expression
def validate_calculator(args):
    expr = args.get("expression", "")
    if not re.fullmatch(r"[0-9+\-*/(). %]+", expr.strip()):
        return False, "expression_not_math"
    # Additional: check balanced parens, valid syntax
    return True, None

# TextToBbox: cardinality check
def validate_textToBbox(args, query):
    top1 = args.get("top1", True)
    if is_singular_query(query) and not top1:
        return False, "cardinality_mismatch"
    if is_plural_query(query) and top1:
        return False, "cardinality_mismatch"
    return True, None
```

#### 3. **Sequence Validation Rules**
```python
def validate_sequence(tool, produced_outputs):
    needs_bbox = tool in {'DrawBox', 'CountGivenObject', 'RegionAttributeDescription'}
    if needs_bbox and 'bbox' not in produced_outputs:
        return False, "missing_bbox_dependency"
    return True, None
```

### Stage 2: ML Model (Learned Patterns)

**Training Data from Analysis**:
- **Positive samples**: 833 correct tool calls (1921 - 1088 invalid)
- **Negative samples**: 
  - Tool mismatch: 137 (with confusion pair labels)
  - Arg mismatch: 85 (with specific error types)
  - Missing tool: 566 (with expected tool labels)

**Feature Engineering**:
- Query embedding (BERT)
- Image features (SSL4EO-S12 pretrained)
- Tool confusion probabilities (learned from 137 mismatch pairs)
- Argument pattern matching (learned from 78 math expression errors)

---

## Impact Projections

### If Constraint Rules Implemented (Conservative)

**Missing Tool (52% of silent errors)**:
- Hard to fix with rules (agent chooses to answer directly)
- Rules can detect "this looks like it needs a tool" heuristically
- **Expected reduction**: 20-30%

**Tool Mismatch (12.6% of silent errors)**:
- Top 4 confusion patterns (66 cases, 48%) addressable by rules
- **Expected reduction**: 40-50%

**Arg Mismatch (7.8% of silent errors)**:
- Calculator math expression: 78/85 cases (91.8%) fixable
- **Expected reduction**: 85-95%

**Sequence Violation (27.6% of silent errors)**:
- Dependency tracking straightforward
- **Expected reduction**: 90-95%

### Overall Silent Failure Reduction
- **Current**: 1,088 / 1,921 = 56.6%
- **With Constraints**: ~680 / 1,921 = 35.4%
- **Improvement**: **21.2 percentage points** reduction in silent failures

### Metric Impact (Estimated)
- **ToolAcc**: +8-12 points (from fixing tool mismatch + some missing tools)
- **ArgAcc**: +3-5 points (from fixing arg mismatch)
- **InstAlign**: +5-8 points (from overall step correctness improvement)

---

## Next Steps

1. **Implement Full Constraint Rules** (extend to all 14 tools)
2. **Run Analysis on More Models** (internlm3, qwen3 variants)
3. **Build Training Dataset** (balanced sampling from gold+error steps)
4. **Train ML Validator** (SSL4EO-S12 + BERT features)
5. **A/B Test Integration** (with/without validator in agent loop)
6. **Iterative Refinement** (analyze remaining errors, add rules)

---

## Code

Analysis script: `/home/james/ThinkGeo/scripts/analyze_invalid_without_error.py`

Usage:
```bash
python scripts/analyze_invalid_without_error.py \
    --pred-files opencompass/outputs/default/MODEL_NAME/predictions/*/ThinkGeo_bench_*.json \
    --detailed
```
