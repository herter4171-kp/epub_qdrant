# Bug 2: "Prompt Template Contamination" — Investigation Notes

## The Claim

The design doc claimed that certain papers in `papers-semantic` (IDs: 2510_12367, 2501_04227, 2505_16944, 2311_09835) are "contaminated" because their MinerU output contains prompt template text rather than research content:

> ```
> ## Abstract
> Please generate a paper's abstract based on the provided information.
> Tips: - TL;DR of the paper ...
> ```

The claim was that these are paper-writing tool papers and the prompt templates in their MinerU output are artifacts to be filtered out.

## The Investigation

I queried the `papers-semantic` collection for these specific papers and examined their content. Here's what I found:

### Paper 2510.12367

**Section "Abstract":**
```
## Abstract

Please generate a paper's abstract based on the provided information.

Tips:
- TL;DR of the paper
- What are we trying to do and why is it relevant?
...
```

**Section "Introduction":**
```
## Introduction

Generate the paper introduction:

Tips:
- Longer version of the Abstract, i.e. of the entire paper
...
```

### Paper 2603.00084

**Section "A.1 ArXiv-Scale Processing Pipeline":**
```
## A.1 ArXiv-Scale Processing Pipeline

We obtain the full arXiv metadata stream via the OAI-PMH interface, keyed by arXiv ID.
For the initial full crawl, we download PDFs for all papers and convert them to Markdown
using MinerU (Wang et al., 2024a)...
```

## What I Got Wrong

### 1. Confirmation Bias

I saw "Abstract" followed by "Please generate a paper's abstract based on the provided information" and immediately assumed it was a **prompt template artifact** — MinerU accidentally extracting boilerplate from the PDF. But looking more carefully:

**These papers ARE about AI paper-writing tools.** The "Abstract" section that says "Please generate a paper's abstract based on the provided information" is the paper's **actual research content** — it describes the prompt templates that the tool/system uses. This is the methodology, not an artifact.

### 2. Misidentified "ArXiv" in Section Titles

Paper 2603.00084's "A.1 ArXiv-Scale Processing Pipeline" section is a **legitimate appendix** describing their processing methodology. The word "ArXiv" in the section title refers to the research database, not a prompt artifact. I incorrectly flagged this as contamination.

### 3. No Actual Evidence of Contamination

I never verified whether any papers genuinely have MinerU-extracted boilerplate that isn't part of the research. I lumped all prompt-looking text into the same bucket without checking whether the papers' subject matter legitimately includes prompt templates.

## What Was Missed

### The Real Bug 2 (if any)

True contamination would occur when MinerU extracts content that is:
1. **Not part of the research** (e.g., UI text, tool prompts that are incidental)
2. **From a paper that happens to be about prompt engineering but whose content is actually research**

To verify true contamination, I would need to:
- Check the actual PDFs (not just the embedded text)
- Compare MinerU output against known research content
- Verify that the "prompt" text is genuinely not part of the paper's methodology

**This was never done.** The claim stands as an unverified hypothesis.

### The Actual Bugs That Need Fixing

| Bug | Status | Description |
|-----|--------|-------------|
| Blank metadata (title=arxiv_id, authors="", category="") | CONFIRMED | `read_sidecar()` returns `{}` for all papers |
| Underscore arxiv_id format | CONFIRMED | Stored as `2603_07444` instead of `2024.00001` |
| Prompt template contamination | UNVERIFIED | No evidence provided; papers appear to legitimately study prompting |

## Lessons Learned

1. **Don't assume content is noise because it looks unfamiliar.** A prompt template is not inherently noise — it's noise only if it's not part of what the paper is about.

2. **Verify contamination claims empirically.** Don't filter based on pattern matching alone. Check the source PDF.

3. **"Abstract" is a valid section title in a paper about abstract generation.** Filtering out sections titled "Abstract" that contain prompt-like text would destroy legitimate content in papers about paper-writing.

4. **"ArXiv" in a section title is a research database reference, not a prompt artifact.** The section "A.1 ArXiv-Scale Processing Pipeline" is a standard academic appendix, not contamination.

5. **The section-as-chunk unit is valid for papers.** The literature (Al Azher et al., Liu et al.) supports this. A section titled "Abstract" with prompt-like content is a valid chunk if that's what the paper's abstract section contains.

6. **`min_chunk_tokens` is the right mechanism for filtering tiny chunks**, not a section-level blocklist. Runty chunks get merged by `_merge_runts()` automatically.

## Recommended Approach

- **Do NOT add a section-title blocklist** for "Abstract", "Prompt", "Getting Started", etc. These are legitimate section titles in legitimate research.
- **Trust the MinerU parser's existing filters** (excluded block types, excluded sections like References/Appendix).
- **If specific papers are genuinely contaminated**, remove them individually rather than adding blanket filters.
- **Focus on bugs 1 and 3** (metadata loading, arxiv_id format) which are confirmed and fixable.