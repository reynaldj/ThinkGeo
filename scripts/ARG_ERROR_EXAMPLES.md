# Real Examples of Argument Errors in Qwen2.5-7B IsInstanceFixed

## 1. expression_not_math (78 cases - 91.8% of arg mismatches)

The Calculator tool expects **pure mathematical expressions** with numbers and operators only. Agent sometimes sends Python code with variable assignments, imports, or complex logic.

### Example 1: Variable Assignment (Most Common Pattern)

**File**: `ThinkGeo_bench_0.json`, Task 11, Step 4

**Ground Truth Expected**:
```python
Calculator(
    expression="round((((1952 - 1360)**2 + (758 - 281)**2)**0.5) * 0.015, 2)"
)
```
This is a simple mathematical expression calculating distance.

**Agent Predicted**:
```python
Calculator(
    expression="distance = ((1952 - 1360) * 0.015)**2 + ((758 - 281) * 0.015)**2, distance ** 0.5"
)
```

**Why It Failed**:
- ❌ Contains variable assignment: `distance = ...`
- ❌ Uses undefined variable: `distance ** 0.5`
- ❌ My validator: `expression_not_math` (contains `=` which is not in `[0-9+\-*/(). %]`)
- ❌ ThinkGeo strict check: Arguments don't match exactly

**Pattern**: Agent is trying to write Python code instead of just providing the math expression.

---

### Example 2: Full Python Program with Imports

**File**: `ThinkGeo_bench_8.json`, Task 32, Step 1

**Ground Truth Expected**:
```python
Calculator(
    expression="(((195+219)/2 - (215+240)/2)**2 + ((772+813)/2 - (772+813)/2)**2)**0.5"
)
```

**Agent Predicted**:
```python
Calculator(
    expression="""from math import sqrt; bbox1 = [3115, 1060, 3150, 1105]; 
                  bbox2 = [3149, 957, 3172, 1000]; 
                  bbox3 = [3181, 1061, 3204, 1112]; 
                  centroid1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]; 
                  centroid2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]; 
                  centroid3 = [(bbox3[0] + bbox3[2]) / 2, (bbox3[1] + bbox3[3]) / 2]; 
                  distances = [sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2), ...];
                  min_distance_pixels = min(distances); 
                  min_distance_meters = min_distance_pixels * 0.125083119972; 
                  min_distance_meters"""
)
```

**Why It Failed**:
- ❌ Contains Python keywords: `from`, `import`, `sqrt`, `min`
- ❌ Contains variable assignments and list comprehensions
- ❌ Contains semicolons
- ❌ My validator: `expression_not_math` (contains letters, brackets)
- ❌ ThinkGeo strict check: Completely different from expected expression

**Pattern**: Agent is trying to write full executable Python programs instead of mathematical expressions.

---

### Example 3: Assignment with Math Functions

**File**: `ThinkGeo_bench_8.json`, Task 37, Step 1

**Ground Truth Expected**:
```python
Calculator(
    expression="(((195+219)/2 - (215+240)/2)**2 + ((772+813)/2 - (772+813)/2)**2)**0.5"
)
```

**Agent Predicted**:
```python
Calculator(
    expression="""import math; x1, y1 = 207, 792.5; x2, y2 = 227.5, 792.5; 
                  distance_pixels = math.sqrt((x2 - x1)**2 + (y2 - y1)**2); 
                  distance_meters = distance_pixels * 0.125779850308; 
                  distance_meters"""
)
```

**Why It Failed**:
- ❌ Contains Python imports: `import math`
- ❌ Uses qualified functions: `math.sqrt()`
- ❌ Multiple variable assignments
- ❌ My validator: `expression_not_math` (contains letters, semicolons)
- ❌ ThinkGeo strict check: Wrong expression entirely

**Pattern**: Agent thinks Calculator is a general-purpose Python evaluator, not realizing it should only handle mathematical expressions.

---

## 2. cardinality_mismatch (7 cases - 8.2% of arg mismatches)

The TextToBbox tool has a `top1` parameter that controls how many bboxes to return:
- `top1=true`: Return **only the best single match**
- `top1=false`: Return **all matching objects**

Agent sometimes gets this backwards based on query wording.

### Example 1: Plural Query → Single Result

**File**: `ThinkGeo_bench_1.json`, Task 4, Step 0

**Ground Truth Expected**:
```python
TextToBbox(
    image="/path/to/image.jpg",
    text="no-damage building",
    top1=True  # ← Requesting SINGLE building
)
```

**Agent Predicted**:
```python
TextToBbox(
    image="/path/to/image.jpg",
    text="no-damage building",
    top1=False  # ← Requesting ALL buildings (WRONG!)
)
```

**Why It Failed**:
- ✅ Same tool, same image, same text query
- ❌ Wrong cardinality: `top1=False` instead of `True`
- **Reason**: Query is singular "no-damage building" → should request single match
- **Agent mistake**: Didn't properly parse the cardinality intent

---

### Example 2: Plural Query Misunderstood

**File**: `ThinkGeo_bench_2.json`, Task 23, Step 1

**Ground Truth Expected**:
```python
TextToBbox(
    image="/path/to/image.jpg",
    text="car",  # ← Query is singular
    top1=False  # ← But gold expects ALL cars
)
```

**Agent Predicted**:
```python
TextToBbox(
    image="/path/to/image.jpg",
    text="car",  # ← Same query
    top1=True   # ← Agent requests ONLY the first car (WRONG!)
)
```

**Why It Failed**:
- ✅ Same tool, image, and text
- ❌ Wrong cardinality: `top1=True` instead of `False`
- **Context**: The task probably needs ALL cars (to count them or analyze all), not just the top one
- **Agent mistake**: Defaulted to `top1=True` without understanding context

---

### Example 3: Modified Query + Wrong Cardinality

**File**: `ThinkGeo_bench_3.json`, Task 9, Step 1

**Ground Truth Expected**:
```python
TextToBbox(
    image="/path/to/image.jpg",
    text="car near red car",  # ← Specific query
    top1=True   # ← Single result
)
```

**Agent Predicted**:
```python
TextToBbox(
    image="/path/to/image.jpg",
    text="car",  # ← Simplified query (lost "near red car")
    top1=False  # ← Requests ALL cars (WRONG!)
)
```

**Why It Failed**:
- ❌ Changed the query (lost specificity)
- ❌ Wrong cardinality: Asks for all instead of the specific one
- **Agent mistakes**: 
  1. Simplified query, losing context
  2. Didn't realize simplified query + `top1=False` would return too many results

---

### Example 4: Vague Query + Wrong Cardinality

**File**: `ThinkGeo_bench_3.json`, Task 24, Step 1

**Ground Truth Expected**:
```python
TextToBbox(
    image="/path/to/image.jpg",
    text="house nearby pool",  # ← Specific combination
    top1=True  # ← Single specific house
)
```

**Agent Predicted**:
```python
TextToBbox(
    image="/path/to/image.jpg",
    text="object",  # ← Generic term (lost specificity!)
    top1=False  # ← Requests ALL objects (DEFINITELY WRONG!)
)
```

**Why It Failed**:
- ❌ Completely changed query to generic "object"
- ❌ Wrong cardinality: `top1=False` would return hundreds of all objects
- **Agent mistakes**: 
  1. Lost all specificity in query
  2. Assumed generic query needs multiple results

---

### Example 5: Plural Query → Single Result

**File**: `ThinkGeo_bench_5.json`, Task 42, Step 1

**Ground Truth Expected**:
```python
TextToBbox(
    image="/path/to/image.jpg",
    text="swimming pools",  # ← Plural (want all pools)
    top1=False  # ← Get ALL swimming pools
)
```

**Agent Predicted**:
```python
TextToBbox(
    image="/path/to/image.jpg",
    text="nearest swimming pool",  # ← Added "nearest" (narrowed to one)
    top1=True  # ← Get only the nearest pool
)
```

**Why It Failed**:
- ❌ Changed query from "swimming pools" (all) to "nearest swimming pool" (one specific)
- ❌ Changed cardinality to match modified query
- **ThinkGeo expected**: All pools to be located
- **Agent logic**: "If I modify the query to ask for 'nearest', then I should use top1=True"
- **The problem**: Agent shouldn't have modified the query!

---

## Summary: Why These Errors Matter

### expression_not_math Issues:
The agent is **hallucinating Python programming capabilities** into the Calculator tool. It should realize:
- ❌ Calculator ≠ Python REPL
- ❌ Calculator ≠ Code executor
- ✅ Calculator = Simple mathematical expression evaluator

**Validator Fix**: 
```python
# Reject if contains:
# - Semicolons
# - Imports (import, from)
# - Assignments (=)
# - Undefined variables (any letter not in "round", "sqrt", etc.)
# - Python builtins (len, min, max, etc.)
```

### cardinality_mismatch Issues:
The agent **fails to extract cardinality intent** from queries. It should:
- ❌ NOT simplify specific queries
- ❌ NOT add extra context like "nearest"
- ✅ Preserve exact query and infer cardinality from context

**Validator Fix**:
```python
if "all" in query or "every" in query or is_plural(query):
    top1 = False
elif "the " in query or "nearest" in query or is_singular(query):
    top1 = True
else:
    top1 = True  # default
```

These 85 cases (78 + 7) represent errors that are **easily fixable with better validation rules** before calling the tools!
