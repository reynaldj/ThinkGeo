# Semantic Argument Checking: Enhanced Analysis Results

**Update Date**: December 8, 2025  
**Change**: Upgraded from format-only validation to semantic (ground truth) comparison

---

## The Problem We Solved

**Before**: Analysis only caught "obviously broken" arguments (unparseable, missing fields)
- Missed ~180 arguments that looked valid but had wrong values
- Example: `Calculator(expression="10 + 5")` passes format check even if gold expected `"15 + 20"`

**After**: Analysis compares all arguments to ground truth, catching ALL mismatches
- Now detects semantic errors (wrong values) not just format errors
- Aligns with ThinkGeo's strict `arg_acc` metric that uses exact equality

---

## Qwen2.5-7B IsInstanceFixed: Updated Results

### Complete Error Breakdown (1,921 steps)

| Category | Count | % of Total | % of Silent Invalid |
|----------|-------|-----------|-------------------|
| Explicit Error | 212 | 11.0% | — |
| **Silent Invalid** | **1,334** | **69.4%** | 100% |
| └─ Missing Tool | 566 | 29.5% | 42.4% |
| └─ Tool Mismatch | 137 | 7.1% | 10.3% |
| └─ **Arg Mismatch** | **331** | **17.2%** | **24.8%** |
| └─ Sequence Violation | 300 | 15.6% | 22.5% |
| **Correct** | 375 | 19.5% | — |

### Argument Mismatch Detailed Breakdown (331 cases)

#### By Tool

| Tool | Count | % of Arg Errors | Type |
|------|-------|----------------|------|
| **Calculator** | **138** | **41.7%** | Math expressions |
| **TextToBbox** | **28** | **8.5%** | Localization queries |
| **DrawBox** | **7** | **2.1%** | Bbox coordinates |
| **GoogleSearch** | **4** | **1.2%** | Search queries |
| **CountGivenObject** | **4** | **1.2%** | Object counting |
| **Other** | **150** | **45.3%** | Various tools |

#### By Error Type

| Error Type | Count | Meaning |
|-----------|-------|---------|
| **expression_mismatch** | 87 | Valid math but wrong calculation |
| **expression_not_math** | 51 | Invalid format (Python code, variables, etc.) |
| **text_mismatch** | 18 | Wrong query text in TextToBbox |
| **bbox_mismatch** | 7 | Valid format but wrong coordinates |
| **top1_mismatch** | 8 | Wrong cardinality (single vs multiple) |
| **Other** | 160 | Various parameter mismatches |

---

## What This Means

### 1. **Calculator Dominates Arg Errors (41.7%)**

**Split**:
- **87 expression_mismatch**: Agent sends valid math but computes wrong value
  - Example: Gold `"15 + 20"`, Agent `"10 + 5"` (both valid expressions, wrong result)
- **51 expression_not_math**: Agent sends Python code instead of math
  - Example: Gold `"10 + 5"`, Agent `"distance = calc(); distance ** 0.5"`

**Impact**: 138/331 = 41.7% of all argument errors are Calculator-related

### 2. **TextToBbox Arg Errors (8.5%)**

**Split**:
- **18 text_mismatch**: Agent queries wrong object
  - Gold: "red car", Agent: "blue car"
- **8 top1_mismatch**: Agent requests wrong cardinality
  - Gold: `top1=True` (single), Agent: `top1=False` (all)
- **2 image_mismatch**: Agent uses different image

**Impact**: Only 28/331 (8.5%) but highly semantically important for localization

### 3. **DrawBox Arg Errors (2.1%)**

**7 bbox_mismatch**: Valid format but wrong coordinates
- Gold: `[10, 20, 100, 200]`
- Agent: `[15, 25, 105, 205]` ← Close but not exact!

---

## Comparison: Before vs After Semantic Checking

### Silent Invalid Error Breakdown

**Before (Format-Only)**:
```
arg_mismatch: 85 cases
├─ expression_not_math: 78
└─ top1_mismatch: 7
Hidden semantic errors: ~180 (not detected)
```

**After (Semantic)**:
```
arg_mismatch: 331 cases (3.9x increase!)
├─ expression_mismatch: 87 (NEW - valid format, wrong value)
├─ expression_not_math: 51 (retained)
├─ text_mismatch: 18 (NEW - wrong query)
├─ top1_mismatch: 8 (renamed from cardinality_mismatch)
├─ bbox_mismatch: 7 (NEW - valid format, wrong coordinates)
└─ Other: 160 (NEW - various parameters)
```

### Impact on Total Silent Invalid Rate

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total Silent Invalid | 1,088 | 1,334 | +246 (+22.6%) |
| Silent Invalid Rate | 56.6% | 69.4% | +12.8% |
| Arg Mismatch Count | 85 | 331 | +3.9x |
| Arg Mismatch % | 7.8% | 24.8% | +3.2x |

---

## Alignment with ThinkGeo Metrics

### ThinkGeo arg_acc Calculation

```python
arg_acc = (steps with correct tool AND correct args) / (total tool steps) * 100
        = 183 / 970 * 100
        = 18.9%
```

This means: **81.1% of tool calls have at least one argument wrong**

### Our Analysis Finds

In the 1,334 silent invalid cases:
- **331 have argument errors** = 24.8% of silent invalids
- Breakdown shows these are mostly:
  - 87 wrong calculations (valid syntax but wrong result)
  - 51 invalid Python code instead of math
  - 18 wrong queries
  - 7 wrong coordinates

**Connection**: ThinkGeo's strict `arg_acc` requires EXACT equality, so even small differences fail. Our semantic analysis explains WHERE these failures happen.

---

## Implementation Improvements

### What Changed in analyze_invalid_without_error.py

**Old Strategy** (Format-Only):
```python
def _arg_check(tool, args, gold_args):
    # Check if args LOOK valid
    if tool == "Calculator":
        if _is_math(args["expression"]):  # passes if it looks like math
            return True
    # Misses: valid-looking math with wrong calculations
```

**New Strategy** (Semantic):
```python
def _arg_check(tool, args, gold_args):
    # First: check exact equality
    if args == gold_args:
        return True, []
    
    # If different, identify specifically how they differ
    if tool == "Calculator":
        if args["expression"] != gold_args["expression"]:
            if not _is_math(args["expression"]):
                error = "expression_not_math"
            else:
                error = "expression_mismatch"  # NEW
    
    return False, [error]
```

**Result**: Catches ALL 331 argument mismatches, not just the 85 format errors

---

## Tools Coverage

The semantic checker now provides detailed breakdowns for all 14 ThinkGeo tools:

| Tool | Checks |
|------|--------|
| TextToBbox | image, text, top1 |
| Calculator | expression (with detailed error types) |
| DrawBox | bbox |
| CountGivenObject | image, name |
| RegionAttributeDescription | image, region |
| ImageDescription | image |
| GetObjects | image |
| ObjectsDistance | object1, object2 |
| GetImageURL | image |
| ChangeDetection | image1, image2 |
| SegmentObjectPixels | image, object_name |
| Solver | equation |
| Plot/OCR/AddText/GoogleSearch | all parameters |

---

## Next Steps

1. **Validator Implementation** should now account for all 331 cases:
   - 87 expression_mismatch: Teach correct calculations
   - 51 expression_not_math: Reject Python code
   - 18 text_mismatch: Better query generation
   - Others: Parameter-specific validation

2. **ML Training Data**: Now have 331 semantic error examples vs 85 before
   - 3.9x more training signal for semantic errors
   - Better ground truth for what NOT to do

3. **Metrics Tracking**: This explains the 81.1% arg failure rate in ThinkGeo
   - Clear breakdown of where errors occur
   - Prioritization for validator rules

---

## Files

- Analysis script: `/home/james/ThinkGeo/scripts/analyze_invalid_without_error.py` (UPDATED)
- Results files: `/home/james/ThinkGeo/scripts/analysis_*.txt` (all 6 models)
