# PCIB Detector Research Paper

This directory contains the LaTeX source for the empirical paper on hallucination detection using Predictive Coding and Information Bottleneck principles.

## Files

- `main.tex` - Main paper source (complete, ready for compilation)
- `references.bib` - Bibliography with all cited works
- `README.md` - This file

## Paper Contents

The paper describes the exact science implemented in the `pcib_detector` codebase:

### Method (Sections 3-3.7)
1. **Claim Extraction**: Atomic factual claim decomposition via LLM with structured output
2. **Verification**: 4-step Chain-of-Thought prompting with probability distributions over {ENTAIL, CONTRADICT, UNKNOWN}
3. **Signal 1 - Evidence Uptake**: `U = KL(p(y|e) || p_0(y))` where p_0 is uniform
4. **Signal 2 - Bottleneck Stress**: `S = JS(p(y|e), p(y|e+distractor))`
5. **Signal 3 - Conflict Sensitivity**: `C = JS(p(y|e), p(y|e+conflict))`
6. **Signal 4 - Trace Validation**: Forward/backward trace comparison with rationalization detection
7. **Scoring Function**: `s = 3.0*contradict + 2.0*(1-entail) + 0.5*(0.8*(1-consistency) + 0.6*(1-support) + 1.2*rationalization)`

### Experiments (Section 4)
- **Dataset**: PatronusAI/HaluBench, 600 examples (first 600, no stratification)
- **Model**: gpt-4o-mini as verifier
- **Metrics**: AUROC, AUPRC, F1 with 1000-sample bootstrap CIs (95%)
- **Ablations**: 6 configurations tested
  - Baseline (PC+IB only)
  - PCIB+Traces (full system)
  - PCIB+Traces (no rationalization)
  - Ensemble (n=1, 3, 5)

### Results (Section 4.3)
Tables are provided with placeholder values. After running the ablation study, results can be copy-pasted from `ablation_results/`:
- `table_ablations.tex` - Main results table
- `table_baselines.tex` - Baseline comparisons
- `metrics.json` - Complete numerical results

## Compiling the Paper

### Requirements

- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages: `amsmath`, `graphicx`, `booktabs`, `natbib`, `hyperref`, `algorithm`, `algorithmic`

### Compilation Commands

```bash
cd paper

# Standard compilation
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Or use latexmk for automatic compilation
latexmk -pdf main.tex

# Clean auxiliary files
latexmk -c
```

### Output

The compilation produces `main.pdf` with the complete paper.

## Experimental Results

**IMPORTANT**: The tables in the paper currently contain placeholder/estimated values. To get actual experimental results:

1. Run the ablation study:
   ```bash
   cd ..  # Return to project root
   export OPENAI_API_KEY=sk-...
   python ablation_study.py --limit 600 --model gpt-4o-mini --no-stratify
   ```

2. After completion (approximately 30-60 minutes with parallel execution), results will be in `ablation_results/`:
   - `metrics.json` - All numerical metrics with CIs
   - `table_ablations.tex` - LaTeX table ready for paper
   - `executive_summary.txt` - Key findings summary
   - `figure_*.pdf` - Plots (if matplotlib is installed)

3. Replace placeholder tables in `main.tex` with actual results from `table_ablations.tex`

## Paper Structure

- **Abstract**: Summary of approach and key results
- **Section 1 (Introduction)**: Motivation, contributions
- **Section 2 (Related Work)**: Prior work on hallucination detection, predictive coding, information bottleneck
- **Section 3 (Method)**: Complete technical description matching codebase
  - 3.1: Problem formulation
  - 3.2: Claim extraction
  - 3.3: Verification
  - 3.4-3.7: Four detection signals
  - 3.8: Scoring function
  - 3.9: Implementation
- **Section 4 (Experiments)**: Dataset, protocol, results, ablations, error analysis
- **Section 5 (Discussion)**: Theoretical insights, practical deployment, limitations, future work
- **Section 6 (Related Work Extended)**: Detailed related work
- **Section 7 (Conclusion)**: Summary and impact
- **Appendix A**: Hyperparameters
- **Appendix B**: Complete prompts
- **Appendix C**: Additional results

## Notes

### Science Accuracy

The paper **exactly matches** the science in the codebase. Every equation, weight, and hyperparameter corresponds to the actual implementation in:
- `pcib_detector/src/pcib_detector/core.py` (lines 362-461)
- `pcib_detector/src/pcib_detector/trace_validation.py` (lines 309-344)
- `pcib_detector/src/pcib_detector/math_utils.py`
- `ablation_study.py` (experimental setup)

### No Stratification

The paper reflects the actual experimental setup used: `--no-stratify` flag, meaning sequential sampling of the first 600 examples without balancing by answer length. The original draft mentioned stratification but has been corrected.

### Reproducibility

The paper includes complete reproducibility information:
- Exact command to reproduce results
- Hyperparameters table in Appendix A
- Complete prompts in Appendix B
- Run ID and timestamp tracking

## Citation Format

```bibtex
@article{pcib2024,
  title={Detecting Hallucinations in Large Language Models via Predictive Coding and Information Bottleneck},
  author={Anonymous},
  year={2024},
  note={Under review}
}
```

## Contact

For questions about the paper or experiments, refer to the main project documentation in the parent directory.
