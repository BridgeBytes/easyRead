Below is the **extracted and structured Terms of Reference (ToR)** converted into a **developer-friendly Markdown format**.

This version is structured to:

* ✅ Break down deliverables into **billable work packages**
* ✅ Map tasks to **technical activities**
* ✅ Align with **GitHub repo activities**
* ✅ Justify **billable hours for invoice**
* ✅ Support milestone-based payment structure (50% / 50%)

---

# 📄 Terms of Reference (ToR)

## Prototyping of AI-Enabled Easy Read Translation Service – Phase II

**Client:** Deutsche Gesellschaft für Internationale Zusammenarbeit (GIZ) GmbH
**Unit:** Digital Transformation Center Rwanda – AI Hub
**Project Duration:** 1 December 2025 – 28 February 2026
**Total Level of Effort:** 32 Expert Days

---

# 1. 🧠 Project Overview

The project aims to develop a **scalable AI-enabled Easy Read translation system** leveraging:

* Large Language Models (LLMs)
* Human-in-the-loop validation
* NLP pipelines
* Image matching
* Text-to-Speech (TTS)
* Accessibility compliance (WCAG)

The system must merge two existing prototypes into a single production-ready MVP.

---

# 2. 🎯 Scope of Work (Billable Work Packages)

---

## 🔹 WP1: Inception & Technical Planning

**Estimated Effort:** 3–4 Days

### Activities

* Review Prototype A (Translation + TTS)
* Review Prototype B (Simplification + Image Matching)
* Analyze architecture and dependencies
* Define final technical stack
* Define LLM strategy (OpenAI, Anthropic, Stability AI APIs)
* Risk assessment
* Define migration roadmap
* Define branching strategy
* Initialize Git repository
* Setup project management structure

### GitHub Deliverables

* Repository initialized
* Branching model defined (e.g., GitFlow)
* README with system architecture draft
* Initial project board setup
* Inception Report (Markdown/PDF)

### Acceptance Criteria

* Approved Inception Report
* Confirmed migration roadmap
* Repository operational

---

## 🔹 WP2: Infrastructure Migration & DevOps

**Estimated Effort:** 6–7 Days

### Activities

* Containerization using Docker
* Kubernetes deployment planning (if required)
* Migration to Hugging Face infrastructure
* Environment setup (dev/staging)
* CI/CD pipeline setup (GitHub Actions)
* Automated build triggers
* Dependency locking
* Environment variable management
* Logging & monitoring setup

### GitHub Deliverables

* Dockerfile
* docker-compose.yml
* CI/CD pipeline (.github/workflows)
* Build validation logs
* Deployment documentation

### Acceptance Criteria

* System runs on Hugging Face
* Automated builds triggered successfully
* Reproducible environment

---

## 🔹 WP3: Core MVP Integration (Backend Engineering)

**Estimated Effort:** 8–9 Days

### Activities

* Merge:

  * Text Simplification
  * Translation
  * Text-to-Speech
  * Image Matching
* Create unified API layer
* Optimize prompt engineering
* Reduce LLM latency
* API integrations:

  * OpenAI
  * Anthropic
  * Stability AI
* Error handling & fallback logic
* Pipeline orchestration

### GitHub Deliverables

* Unified backend service
* Integrated pipeline module
* API abstraction layer
* Prompt templates
* End-to-end integration test scripts

### Acceptance Criteria

* Single text input produces:

  * Simplified text
  * Audio output
  * Matching image
* No pipeline errors
* Context alignment validated

---

## 🔹 WP4: Playground UI Development

**Estimated Effort:** 6–7 Days

### Activities

* Develop frontend playground interface
* Side-by-side model comparison
* Real-time readability scoring
* Feedback capture module
* UI accessibility compliance
* API connection to backend
* Performance optimization

### GitHub Deliverables

* Frontend repository/module
* UI components
* Readability scoring implementation
* Feedback logging system

### Acceptance Criteria

* Non-technical user usability
* Real-time readability metrics
* Functional model comparison
* Responsive UI

---

## 🔹 WP5: Quality Assurance & Accessibility Validation

**Estimated Effort:** 3–4 Days

### Activities

* Performance evaluation
* Readability metric validation
* WCAG compliance testing
* Cognitive accessibility checks
* Bug fixing
* Regression testing
* Latency testing

### GitHub Deliverables

* Test cases
* QA report
* Accessibility validation checklist
* Performance benchmarks

### Acceptance Criteria

* Meets custom readability metrics
* Meets WCAG accessibility standards
* Stable MVP

---

## 🔹 WP6: Future-Proofing & ETH Zurich Integration Plan

**Estimated Effort:** 2–3 Days

### Activities

* Design integration pathway for ETH Zurich image creator
* Define extension API contracts
* Modular architecture documentation
* Open-source contribution guidelines
* Scalability roadmap

### GitHub Deliverables

* Integration specification
* Architecture diagrams
* Extension guide
* Contribution guide (CONTRIBUTING.md)

### Acceptance Criteria

* Third-party dev can extend system
* ETH Zurich module integration plan documented

---

## 🔹 WP7: Documentation & Handover

**Estimated Effort:** 2–3 Days

### Activities

* Technical documentation (architecture + APIs)
* Deployment guide
* Developer onboarding guide
* User manual
* Final technical report
* Repository transfer
* Knowledge transfer session

### GitHub Deliverables

* Complete README
* API documentation
* Setup instructions
* Final report
* All repos transferred

### Acceptance Criteria

* Third-party developer can set up within 4 hours
* Final report accepted
* Code assets transferred

---

# 3. 💰 Payment Milestones (Invoice Justification)

| Milestone            | Deliverables                                             | Days | Payment |
| -------------------- | -------------------------------------------------------- | ---- | ------- |
| **Midterm (Week 6)** | Backend merged + Migration complete + Inception approved | 15   | 50%     |
| **Final Delivery**   | Playground UI + Documentation + ETH Plan + Final Report  | 17   | 50%     |
| **Total**            | Full MVP Delivered                                       | 32   | 100%    |

---

# 4. 📊 Total Effort Breakdown (For Invoice Justification)

| Work Package                   | Estimated Days |
| ------------------------------ | -------------- |
| WP1 – Inception & Planning     | 4              |
| WP2 – Infrastructure Migration | 7              |
| WP3 – Core MVP Integration     | 9              |
| WP4 – Playground UI            | 7              |
| WP5 – QA & Accessibility       | 3              |
| WP6 – Future Strategy          | 1              |
| WP7 – Documentation & Handover | 1              |
| **Total**                      | **32 Days**    |

---

# 5. 👤 Required Expert Profile

* Bachelor’s degree in CS / Software Engineering / ML
* 5+ years software engineering experience
* 4+ years AI prototype development (Language AI focus)
* 3 years experience in Rwanda or East Africa
* Business fluency in English (C1)

---

# 6. 🌱 Sustainability Requirements

* Economy-class travel
* Prefer train/e-mobility for short distances
* CO2 offset required for flights
* Travel must be pre-approved

---

# 7. 📦 Final Deliverables Summary

* ✅ Hugging Face deployed system
* ✅ Containerized architecture
* ✅ CI/CD automation
* ✅ Unified AI pipeline
* ✅ Interactive playground UI
* ✅ WCAG-compliant outputs
* ✅ ETH Zurich integration plan
* ✅ Full open-source documentation
* ✅ Final technical report
* ✅ Repository transfer

---

If you'd like, I can next:

* 🔢 Convert this into an **itemized invoice template (with hourly rate modeling)**
* 📁 Convert into a **GitHub Project Board structure**
* 📊 Convert into a **Gantt timeline**
* 🧾 Create a **professional invoice draft aligned to the GIZ contract**
* 📈 Add risk buffer justification for additional billable scope**

Just tell me which one you want.
