# Comparative Analysis: Invalid Tool Call Patterns Across Models

**Analysis Date**: December 8, 2025  
**Dataset**: ThinkGeo Benchmark (1,921 steps across 486 tasks)  
**Mode**: every_with_gt (step-by-step with ground truth context)

---

## Executive Summary

| Model | Invalid w/o Error Rate | Missing Tool | Tool Mismatch | Arg Mismatch | Sequence Violation | Explicit Error |
|-------|----------------------|--------------|---------------|--------------|-------------------|----------------|
| **Qwen1.5 Base** | 64.3% | 1,236 | 0 | 0 | 0 | 311 (16.2%) |
| **Qwen1.5 Fixed** | 67.4% | 1,242 | 8 | 2 | 42 | 231 (12.0%) |
| **Qwen2.5 Base** | 32.6% | 626 | 0 | 0 | 0 | 908 (47.3%) |
| **Qwen2.5 Fixed** | 56.6% | 566 | 137 | 85 | 300 | 212 (11.0%) |
| **Qwen3 Base** | 11.5% | 221 | 0 | 0 | 0 | 1,404 (73.1%) |
| **Qwen3 Fixed** | 15.3% | 224 | 8 | 1 | 61 | 1,291 (67.2%) |

---

## Key Insights

### 1. **Model Generation Evolution: Qwen1.5 → Qwen2.5 → Qwen3**

**Silent Failure Reduction**:
- Qwen1.5 Base: 64.3% → Qwen2.5 Base: 32.6% → Qwen3 Base: 11.5%
- **52.8 percentage point improvement** from Qwen1.5 to Qwen3

**Failure Mode Shift**:
- Qwen1.5: Silent failures (64.3%) >> Explicit errors (16.2%)
- Qwen2.5: Balanced - explicit errors (47.3%) > silent failures (32.6%)
- Qwen3: Explicit errors dominant (73.1%) >> Silent failures (11.5%)

**Interpretation**: Newer models fail more explicitly (raise errors) rather than silently producing wrong outputs. This is actually desirable - explicit errors are easier to handle than silent invalids.

### 2. **isinstance() Fix Impact ("Fixed" versions)**

The isinstance fix paradoxically **increases** silent invalid rates in some cases:

| Model | Base Silent Invalid | Fixed Silent Invalid | Change |
|-------|-------------------|---------------------|--------|
| Qwen1.5 | 64.3% | 67.4% | **+3.1%** ⬆️ |
| Qwen2.5 | 32.6% | 56.6% | **+24.0%** ⬆️ |
| Qwen3 | 11.5% | 15.3% | **+3.8%** ⬆️ |

**Why the increase?**

**Base versions**: Tools fail with `TypeError: arg 2 must be a type` → counted as **explicit_error**

**Fixed versions**: Tools now execute but with wrong parameters → counted as **arg_mismatch** or pass validation but produce wrong outputs

**The Trade-off**:
- ✅ Tools actually run (no crashes)
- ❌ More silent failures (wrong args accepted)
- 🎯 This actually validates the need for our validator!

### 3. **Error Distribution by Model**

#### Qwen1.5-7B BaseStepByStep
```
Explicit Error:    311 (16.2%)  ████
Silent Invalid:  1,236 (64.3%)  █████████████
  - Missing Tool: 100%
  - Tool Mismatch: 0%
  - Arg Mismatch: 0%
  - Seq Violation: 0%
```

**Pattern**: Pure "missing tool" problem - agent gives direct answers instead of using tools.

#### Qwen1.5-7B IsInstanceFixed
```
Explicit Error:    231 (12.0%)  ███
Silent Invalid:  1,294 (67.4%)  ██████████████
  - Missing Tool: 96.0%
  - Tool Mismatch: 0.6%
  - Arg Mismatch: 0.2%
  - Seq Violation: 3.2%
```

**Pattern**: Still dominated by missing tools. isinstance fix slightly worsened the problem.

#### Qwen2.5-7B BaseStepByStep
```
Explicit Error:    908 (47.3%)  ██████████
Silent Invalid:    626 (32.6%)  ███████
  - Missing Tool: 100%
  - Tool Mismatch: 0%
  - Arg Mismatch: 0%
  - Seq Violation: 0%
```

**Pattern**: Major improvement! Explicit errors dominate. When silent failures occur, still all missing tools.

#### Qwen2.5-7B IsInstanceFixed ⭐ **Most Interesting**
```
Explicit Error:    212 (11.0%)  ██
Silent Invalid:  1,088 (56.6%)  ████████████
  - Missing Tool: 52.0%
  - Tool Mismatch: 12.6%
  - Arg Mismatch: 7.8%
  - Seq Violation: 27.6%
```

**Pattern**: **Diversified error modes**! This is the only version showing significant tool confusion (137 cases) and arg errors (85 cases). The isinstance fix allows tools to run, exposing deeper reasoning errors.

#### Qwen3-8B BaseStepByStep
```
Explicit Error:  1,404 (73.1%)  ███████████████
Silent Invalid:    221 (11.5%)  ██
  - Missing Tool: 100%
  - Tool Mismatch: 0%
  - Arg Mismatch: 0%
  - Seq Violation: 0%
```

**Pattern**: Best base model! Only 11.5% silent failures. Fails explicitly when confused.

#### Qwen3-8B IsInstanceFixed
```
Explicit Error:  1,291 (67.2%)  █████████████
Silent Invalid:    294 (15.3%)  ███
  - Missing Tool: 76.2%
  - Tool Mismatch: 2.7%
  - Arg Mismatch: 0.3%
  - Seq Violation: 20.7%
```

**Pattern**: Still excellent. isinstance fix slightly increases silent failures but remains low overall.

---

## Tool Confusion Patterns (Qwen2.5 IsInstanceFixed)

Only Qwen2.5 IsInstanceFixed shows significant tool mismatch (137 cases). Here are the top confusions:

### Top 10 Mismatches
| Expected Tool | Predicted Tool | Count | % of Mismatches |
|---------------|---------------|-------|----------------|
| Solver → Calculator | 24 | 17.5% |
| Calculator → CountGivenObject | 21 | 15.3% |
| TextToBbox → Calculator | 11 | 8.0% |
| Calculator → RegionAttributeDescription | 11 | 8.0% |
| RegionAttributeDescription → DrawBox | 10 | 7.3% |
| Calculator → TextToBbox | 6 | 4.4% |
| TextToBbox → CountGivenObject | 6 | 4.4% |
| RegionAttributeDescription → Calculator | 6 | 4.4% |
| RegionAttributeDescription → TextToBbox | 5 | 3.6% |
| Calculator → Plot | 4 | 2.9% |

**Key Confusion Categories**:
1. **Math Tools**: Solver ↔ Calculator (24 + 0 = 24 cases)
2. **Calculation vs Counting**: Calculator ↔ CountGivenObject (21 + 4 = 25 cases)
3. **Localization vs Calculation**: TextToBbox ↔ Calculator (11 + 6 = 17 cases)
4. **Description vs Visualization**: RegionAttributeDescription → DrawBox (10 cases)

---

## Argument Error Patterns (Qwen2.5 IsInstanceFixed)

### Calculator: expression_not_math (78/85 = 91.8% of arg errors)

Agent passes non-mathematical expressions:
- Natural language: "the number of buildings"
- Incomplete expressions: "distance between"
- Invalid syntax: "count(x) + 5"

### TextToBbox: cardinality_mismatch (7/85 = 8.2% of arg errors)

Agent requests wrong number of results:
- Gold expects single bbox (`top1: true`) → Pred asks for multiple (`top1: false`)
- Gold expects all matches (`top1: false`) → Pred asks for top only (`top1: true`)

---

## Missing Tool Analysis

### Qwen1.5-7B BaseStepByStep - Top Missing Tools
| Tool | Count | % of Missing |
|------|-------|-------------|
| Calculator | 334 | 27.0% |
| TextToBbox | 260 | 21.0% |
| RegionAttributeDescription | 235 | 19.0% |
| CountGivenObject | 116 | 9.4% |
| ChangeDetection | 76 | 6.1% |

### Qwen2.5-7B IsInstanceFixed - Top Missing Tools
| Tool | Count | % of Missing |
|------|-------|-------------|
| TextToBbox | 181 | 32.0% |
| Calculator | 113 | 20.0% |
| RegionAttributeDescription | 51 | 9.0% |
| ChangeDetection | 48 | 8.5% |
| CountGivenObject | 33 | 5.8% |

### Qwen3-8B BaseStepByStep - Top Missing Tools
| Tool | Count | % of Missing |
|------|-------|-------------|
| TextToBbox | 76 | 34.4% |
| Calculator | 47 | 21.3% |
| RegionAttributeDescription | 28 | 12.7% |
| CountGivenObject | 14 | 6.3% |
| DrawBox | 12 | 5.4% |

**Pattern Across Models**: TextToBbox and Calculator are consistently the most missed tools (45-55% combined).

---

## Sequence Violation Patterns

Sequence violations occur when tools requiring outputs from previous steps are called without those dependencies being met.

| Model | Seq Violations | % of Silent Invalid | Notes |
|-------|---------------|-------------------|-------|
| Qwen1.5 Base | 0 | 0% | No violations |
| Qwen1.5 Fixed | 42 | 3.2% | Minor issue |
| Qwen2.5 Base | 0 | 0% | No violations |
| **Qwen2.5 Fixed** | **300** | **27.6%** | Major issue |
| Qwen3 Base | 0 | 0% | No violations |
| Qwen3 Fixed | 61 | 20.7% | Significant issue |

**Key Insight**: Sequence violations appear primarily in "Fixed" versions, suggesting the isinstance fix allows tools to execute but the agent doesn't properly track or utilize outputs from previous steps.

---

## Validator Design Priorities (Based on Qwen2.5 Fixed Analysis)

### High Priority (Impact > 10% of silent failures)

1. **Missing Tool Detection (52% of silent invalid)**
   - Pattern: Agent provides direct answer instead of tool call
   - Strategy: Heuristic rules to detect when a tool call is needed
   - Example: "distance between X and Y" → requires Calculator + TextToBbox

2. **Sequence Violation Prevention (27.6% of silent invalid)**
   - Pattern: Tool calls reference outputs not yet produced
   - Strategy: Dependency tracking across conversation history
   - Example: DrawBox(bbox) requires prior TextToBbox/ObjectDetection

3. **Tool Confusion Resolution (12.6% of silent invalid)**
   - Pattern: Wrong tool selected for task
   - Strategy: Constraint rules for top confusion pairs
   - Target: Solver/Calculator, Calculator/CountGivenObject, TextToBbox/Calculator

### Medium Priority (Impact 5-10% of silent failures)

4. **Argument Validation (7.8% of silent invalid)**
   - Pattern: Calculator gets non-math expressions (91.8% of arg errors)
   - Strategy: Strict regex validation for Calculator.expression
   - Secondary: TextToBbox cardinality detection

### Implementation Recommendation

**Phase 1**: Implement rules for top 4 tool confusions (66 cases, 48% of mismatches)
- Solver vs Calculator
- Calculator vs CountGivenObject  
- TextToBbox vs Calculator
- RegionAttributeDescription vs DrawBox

**Phase 2**: Add argument validators for Calculator and TextToBbox (85 cases total)

**Phase 3**: Implement sequence tracking for bbox dependencies (300 cases)

**Phase 4**: ML model for missing tool detection (566 cases) - hardest problem

---

## Recommendations by Model

### For Qwen1.5 Users
- **Issue**: 64-67% silent failures, all missing tools
- **Solution**: Strong prompt engineering to enforce tool usage
- **Validator Impact**: Limited - need better base prompting first

### For Qwen2.5 Users ⭐ **Best Validator Target**
- **Issue**: 56.6% silent failures with diverse error modes
- **Solution**: Deploy full validator with all 4 priority areas
- **Validator Impact**: High - could reduce to ~25-30% silent failures

### For Qwen3 Users
- **Issue**: Only 11-15% silent failures, mostly explicit errors
- **Solution**: Focus on error recovery rather than prevention
- **Validator Impact**: Moderate - already performing well

---

## Next Steps

1. ✅ **Completed**: Quantitative analysis across 6 model configurations
2. 📋 **Next**: Implement constraint rules for top 4 tool confusions
3. 📋 **Next**: Add Calculator expression validator (covers 78/85 arg errors)
4. 📋 **Next**: Build sequence dependency tracker
5. 📋 **Next**: Train ML model on Qwen2.5 Fixed data (1,088 invalid samples)
6. 📋 **Next**: A/B test validator integration with Qwen2.5

---

## Files Generated

All detailed analysis results saved to:
- `/home/james/ThinkGeo/scripts/analysis_qwen1.5-7bBaseStepByStep.txt`
- `/home/james/ThinkGeo/scripts/analysis_qwen1.5-7bIsInstanceFixed.txt`
- `/home/james/ThinkGeo/scripts/analysis_qwen2.5-7bBaseStepByStep.txt`
- `/home/james/ThinkGeo/scripts/analysis_qwen2.5-7bIsInstanceFixed.txt`
- `/home/james/ThinkGeo/scripts/analysis_qwen3-8bBaseStepByStep.txt`
- `/home/james/ThinkGeo/scripts/analysis_qwen3-8bIsInstanceFixed.txt`

Analysis script: `/home/james/ThinkGeo/scripts/analyze_invalid_without_error.py`
