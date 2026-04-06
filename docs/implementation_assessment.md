# Implementation Assessment Report
**Project:** EasyRead Together – Phase II Prototype
**Client:** GIZ Digital Transformation Center Rwanda – AI Hub
**Contract Period:** 1 December 2025 – 28 February 2026
**Effective Start Date:** 12 December 2025
**Assessment Date:** 27 February 2026
**Total Budget:** 32 Expert Days

---

## Executive Summary

Solid technical progress has been made, but the project is not yet complete. The core AI pipeline — which takes plain text, simplifies it for accessibility, and pairs sentences with supporting icons — is functional and demonstrable. However, several originally planned features (audio output, translation, and automated testing) were not reached within the contract window.

**The primary reason is a delayed start.** The effective start date was 12 December 2025 — twelve calendar days after the contracted start of 1 December. This was a combination of onboarding steps and early project setup, and the resulting eight-working-day shortfall was not recovered during the contract period.

**A secondary reason is scope depth.** Two technically demanding workstreams — a fine-tuned image model and a custom quality-scoring system — proved necessary for a credible prototype but were not fully scoped in the original Terms of Reference. Addressing them consumed time originally planned for other deliverables.

A one-month extension to 31 March 2026 is requested to close the remaining gaps and deliver a fully handover-ready product.

---

## Days Worked

Work logged here reflects total expert effort — not only coding. Architecture decisions, API research,  and integration testing all precede visible output and are not captured in version-control history.

| Work Package | Period | Summary of Activity | Days |
|---|---|---|---|
| WP1 – Inception & Prototype Reviews | Dec 12 – Jan 10 | Initial setup; reviewed two early prototypes (Translation+Audio and Simplification+Icons) | **5** |
| WP2 – Infrastructure & Hosting | Jan 11 – Jan 26 | Built 4-service hosting environment (backend, UI, file storage, secure web access) | **4** |
| WP3 – Core AI Backend | Jan 27 – Feb 20 | AI simplification pipeline; icon retrieval system; fine-tuned image model; quality metrics | **6** |
| WP4 – User Interface | Jan 28 – Feb 19 | Interactive review screen; editable sentence/icon pairs; DOCX and Markdown export | **4** |
| WP5 – Testing & Evaluation | Feb 26 | Quality-scoring script across six dimensions; evaluation report | **2** |
| WP6 – ETH Zurich: Integration Review | Feb 26 | Reviewed ETH Zurich model paper; assessed integration feasibility | **0.5** |
| WP6 – ETH Zurich: Q&A Preparation | Feb 26 | Structured Q&A and integration notes for handover | **0.5** |
| WP7 – Documentation | Pending | README is a placeholder; full documentation not yet started | **0** |
| **Total** | | | **22 days** |

*Effective working period: 12 December 2025 – 28 February 2026. Days remaining against the 32-day contract: **10 days**.*

---

## Deliverables Status

| Work Package | Contract Days | Days Used | Completion | What Is Missing |
|---|---|---|---|---|
| WP1 – Inception & Planning | 4 | 5 | 90% | Formal inception report; project tracking board |
| WP2 – Infrastructure & Hosting | 7 | 4 | 85% | Automated deployment pipeline |
| WP3 – Core AI Backend | 9 | 6 | 80% | Audio (text-to-speech); Translation (Kinyarwanda / French) |
| WP4 – User Interface | 7 | 4 | 70% | Side-by-side model comparison; readability score display; user feedback capture |
| WP5 – Testing & Quality | 3 | 2 | 55% | Accessibility (WCAG) audit; automated test suite |
| WP6 – ETH Zurich Collaboration | 2 | 1 | 30% | Written integration specification; contributor guide |
| WP7 – Documentation | 2 | 0 | 10% | README; deployment guide; API reference; final technical report |
| **Total** | **34** | **22** | **~70%** | |

---

## What Has Been Built

- **AI simplification pipeline** — Three-stage process (Simplify → Check → Refine) powered by a large language model, producing accessible plain-language output from complex source text
- **Icon retrieval system** — Searches three global symbol libraries (ARASAAC, Mulberry, LDS) and falls back to a fine-tuned image generator when no match is found
- **Fine-tuned image model** — Custom-trained on accessibility imagery; benchmark shows +7.5% improvement over the standard baseline
- **Quality scoring** — Six-dimension evaluation of generated icons (colour palette, edge clarity, visual focus, contrast, line weight, centering)
- **Hosted environment** — Four-service containerised stack running on AWS with SSL encryption and file storage. The system was initially deployed on Hugging Face Spaces but migrated to AWS due to the absence of GPU access on the Hugging Face free tier, which is required for the image generation workstream
- **Review interface** — Practitioners can edit sentence/icon pairs before finalising output
- **Export** — Outputs a formatted Word document (two-column layout with icons alongside text) or Markdown

## What Remains

- Audio output (text-to-speech) so documents can be listened to
- Translation into another language before simplification
- Automated build and deployment pipeline
- Side-by-side model comparison and on-screen readability scores in the UI
- A way for end users to rate and comment on outputs
- Formal accessibility audit against WCAG 2.1 AA standards
- Complete documentation: README, deployment guide, API reference, and final technical report

---

## Note on Prototype Reviews (WP1)

The two early prototype reviews — one for the Translation and Audio prototype, one for the Simplification and Icon-Matching prototype — were carried out directly by the project owner from 12 December 2025 onward. This work predates the version-control record and represents 3–4 days of expert effort under WP1. A written account will be submitted separately for invoice purposes.

---

## Timeline Extension Request

### Why an Extension Is Needed

The contract ran from 1 December 2025 to 28 February 2026. The effective start was 12 December 2025 — a twelve-calendar-day gap partly due to onboarding steps and early setup. That shortfall was not recovered over the course of the project.

Additionally, two workstreams — training a domain-specific image model and building a custom quality-measurement framework — required more depth than originally anticipated and displaced time planned for other deliverables.

Of the 32 contracted expert days, **22 have been used** and **10 days remain within the original budget**. The request is therefore not for additional funding, but simply to extend the contract period by one month so that the remaining 10 days can be applied to close the outstanding work packages.

### Proposed Extension

| | |
|---|---|
| Extension period | 1 March 2026 – 31 March 2026 |
| Additional days or budget requested | **None** |
| Days remaining in original budget | **10 days** (32 contracted − 22 used) |
| Action required | Extend the contract end date to 31 March 2026 |

### Work Plan for Extension Period

| # | Work Package | What Will Be Done | Days |
|---|---|---|---|
| 1 | Backend | Add text-to-speech so exported documents include an audio version | 2 |
| 2 | Backend | Add translation layer (Kinyarwanda / French) upstream of simplification; validate full chain | 1 |
| 3 | Infrastructure | Build automated build-and-deploy pipeline; write deployment guide | 2 |
| 4 | User Interface | Side-by-side model comparison; live readability score; user rating and feedback form (saved to file storage) | 1 |
| 5 | Testing & Quality | Accessibility (WCAG 2.1 AA) audit; fix critical issues; automated test suite for pipeline and API | 1 |
| 6 | Documentation | Complete README, API reference, deployment guide, and final technical report; knowledge-transfer session | 3 |
| | **Total** | | **10 days** |

### What Will Be Deliverable by End of Extension

- Complete end-to-end pipeline: Text input → Translation → Simplification → Icon matching → Audio → Export
- Reproducible, automated deployment that any developer can run from scratch
- Accessibility-compliant interface with user feedback capture and model comparison
- Full documentation meeting the original ToR requirement: a third party able to set up the system within four hours
- Final technical report submitted to GIZ

---

*Repository: github.com/BridgeBytes/easyRead | Branch: eval*
