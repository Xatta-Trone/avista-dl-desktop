# **AVISTA CHAPTER - PLANNING & STRUCTURE GUIDE**

## **Proposed Chapter 7 Structure - Section-by-Section Outline**

The following outline provides the full section structure for Chapter 7, including the key content for each section, the suggested figures/tables, and the approximate paragraph count. Use this as a writing blueprint.

**Table 3. Chapter 7 Section-by-Section Structure - Key Content, Figures, and Writing Notes.**

| **Section**                           | **Title**                                    | **Key Content**                                                                                                                                                                                                                              | **Suggested Figure/Table**                                                              |
| ------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **7.1 Introduction**                  | From Research to Practice: Motivating AVISTA | Bridge from Ch4+Ch5 empirical findings to operational tool. Define AVISTA fully. State RQ4. Preview the chapter.                                                                                                                             | _None_                                                                                  |
| **7.2 Contribution Framing**          | AVISTA as a Dissertation Contribution        | Explicitly position AVISTA as a software+analytical artifact contribution. Distinguish from conceptual framework. State the 5-layer architecture. Map to prior contributions (Ch4 = engine, Ch5 = attribution, Ch7 = system).                | _Table 7.1: AVISTA Contribution Summary_                                                |
| **7.3 System Overview**               | Architecture and Design Philosophy           | 5-layer architecture detail. Design goals: infrastructure-sensitive, interpretable, policy-oriented, scalable to state DOT data environments. Technology stack overview (Python/FastAPI/React or similar).                                   | _Figure 7.1: AVISTA System Architecture Diagram (5-layer)_                              |
| **7.4 Pipeline Foundation**           | The v10 Classification Engine                | How the v10 pipeline (06_Phase2_SAE_Classification_v10_ADAS) maps to the tool. Data ingestion → feature engineering → 7-model ensemble → calibration → SCCS selection. Which model is deployed (MambaAttention, SCCS #1).                    | _Table 7.2: v10 Pipeline → AVISTA Module Mapping_                                       |
| **7.5 Interpretability Module**       | SHAP Attribution Engine                      | How TreeSHAP runs in production for single-record explanations. Multi-method consensus layer. OHE aggregation to 24 base features. Per-class SHAP waterfall outputs.                                                                         | _Screenshot 1: Single-record SHAP waterfall Figure 7.2: Interpretability pipeline flow_ |
| **7.6 Decision Support Applications** | Three Application Modules                    | 7.6.1 ODD Boundary Evaluator 7.6.2 Road Segment Risk Screener 7.6.3 Policy Analytics Dashboard Each described with inputs, processing logic, and output format.                                                                              | _Screenshots 2, 3, 4; Table 7.3: Application Module Summary_                            |
| **7.7 User Interface**                | Web Interface Design and Screens             | Dashboard/landing page. Crash record input form. Probability output panel. SHAP explanation panel. ODD risk profile view. Policy analytics view. Accessibility and practitioner UX considerations.                                           | _Screenshots 1-6 (full UI walkthrough)_                                                 |
| **7.8 Illustrative Use Case**         | End-to-End Scenario: Dallas Urban Corridor   | Walk through a single realistic use case: a crash record from an urban Dallas corridor is submitted, classified as AA with P̂(AA)=0.73, SHAP waterfall identifies Pop_Group+Speed as top drivers, ODD boundary flagged, analyst action taken. | _Table 7.4: Use Case Walk-Through Steps_                                                |
| **7.9 Validation and Reliability**    | How AVISTA Inherits Validation from Ch4      | Calibration ECE scores. Bootstrap CI coverage. Weather/lighting robustness (Tables 4.17-4.18). Out-of-time stability. The tool is as reliable as its underlying MambaAttention engine.                                                       | _Reproduce key Ch4 results here in condensed form_                                      |
| **7.10 Limitations and Future Work**  | Current Scope and Planned Extensions         | Implementation status (research prototype vs. production tool). Real-time CRIS feed not yet connected. MambaAttention SHAP gap. V2I integration roadmap. Multi-state generalisation.                                                         | _None_                                                                                  |
| **7.11 Chapter Summary**              | RQ4 Answer and Chapter Synthesis             | Direct answer to RQ4. Summary of three application modules. Brief on contribution framing. Bridge to Chapter 8 (Conclusions).                                                                                                                | _Table 7.5: RQ4 Answer Summary_                                                         |

### **_2.1 Section 7.1: Introduction (1-2 pages)_**

This section bridges Chapters 4-5 and Chapter 7. It should begin with a single paragraph that restates the empirical challenge: we now have a validated classification engine (MambaAttention, SCCS #1), a four-method XAI attribution suite, and well-validated findings on the features that most strongly distinguish SAE levels - but how does a Texas DOT analyst or NHTSA policy officer actually use these outputs to make a decision? The answer is AVISTA.

The section should then formally state RQ4 in full: 'How can SHAP-based feature attributions and automation-level classification outputs from the AVISTA framework be operationalized into a structured, infrastructure-sensitive decision-support framework for ODD boundary evaluation, automation-level risk screening, and AV safety policy development?' and briefly preview the three applications that answer it.

_Writing tip: Start the chapter with a short vignette - a hypothetical scenario of a TxDOT engineer reviewing a new crash record from an urban Dallas intersection and wanting to know whether the vehicle may have been operating at Level 3+ automation. This grounds the chapter in a concrete use case before the technical description begins._

### **_2.2 Section 7.2: AVISTA as a Dissertation Contribution (1-2 pages)_**

This is the most strategically important section of the chapter. Its purpose is to explicitly and formally position AVISTA as a dissertation contribution - not just an application, not just a conceptual framework, but a deliverable artifact that extends the state of the art in AV safety analytics tooling.

The section should argue three points. First, no prior tool integrates TDL-based SAE automation level classification with four-method XAI attribution in a practitioner-accessible web interface - AVISTA is the first such system. Second, AVISTA is domain-specific in a way that general-purpose explainability tools (like the SHAP library's own dashboard) are not: it is designed for Texas CRIS data, with the feature engineering pipeline, VIN ADAS enrichment, and GroupShuffleSplit validation strategy built in. Third, the tool's three-application architecture (ODD evaluation, risk screening, policy analytics) maps directly to the three primary decision-making needs of state DOT crash safety analysts.

**Table 5. AVISTA Dissertation Contributions - Five Contribution Types and Evidence Basis.**

| **Contribution Type**                                 | **What AVISTA Contributes**                                                                                                                                                                                                                                                                          | **Evidence Basis in Dissertation**                                                         |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Novel Methodological Framework**                    | AVISTA is the first infrastructure-sensitive TDL-based SAE automation level classification framework applied to police-reported crash data. It integrates seven TDL architectures, SMOTE-NC class balancing, SCCS model selection, and four-method XAI attribution into a unified analytical system. | _Abstract contribution claim #1; Chapter 4 (classification engine); Chapter 5 (XAI layer)_ |
| **Validated Classification Engine**                   | MambaAttention selected as deployment model via SCCS (#1, 0.5205) with BCa bootstrap CIs confirming superiority over XGBoost and RF. Temporal and infrastructure-type robustness validated via OOT testing.                                                                                          | _Chapter 4 Tables 4.13-4.22; SCCS technical overview; bootstrap CI analysis_               |
| **Four-Method XAI Attribution Suite**                 | First four-method cross-architecture XAI analysis for SAE automation level classification: TreeSHAP, IG, DeepLift, Permutation. Four-way consensus identifies Pop_Group, Speed, Weather, Obj_Struck as definitive discriminators.                                                                    | _Chapter 5 Tables 5.2-5.6; Spearman correlation analysis; per-class beeswarm plots_        |
| **Operational Decision Support Tool (AVISTA System)** | Web-based prototype operationalising the v10 classification pipeline for three practitioner applications: ODD boundary evaluation, road segment risk screening, AV safety policy analytics. Designed for TxDOT/NHTSA deployment environments.                                                        | _Chapter 7 (this chapter); Abstract contribution #4_                                       |
| **Policy-Relevant Safety Evidence**                   | Quantitative evidence that urbanisation (Pop_Group), speed limit, weather, and object struck are the primary SAE class discriminators, validated across four methods and two model architectures. Directly usable for ODD regulatory standards.                                                      | _Chapter 5 Section 5.7 (AVISTA applications); Chapter 4 Tables 4.17-4.18 (robustness)_     |

### **_2.3 Section 7.3: System Overview and Architecture (2-3 pages)_**

This section describes the five-layer architecture in full system terms - going beyond the conceptual description in Chapter 5 (Section 5.7) to include the technology stack, data flow, and implementation notes. The architecture should be presented as a system design diagram (Figure 7.1 - this is the most important figure in the chapter and should be created as a proper diagram showing all five layers with data flow arrows).

Layer 1 (Data Ingestion): accepts new CRIS crash records in CSV or API format; applies the 29-column schema harmonisation pipeline and VIN ADAS enrichment. Layer 2 (Feature Engineering): runs the four-stage selection pipeline (Cramér's V, VIF, domain exclusions, ADAS derivation) to produce the 24-feature input tensor. Layer 3 (Classification Engine): runs MambaAttention (SCCS #1) to produce calibrated P̂(AA|x), P̂(PA|x), P̂(AD|x). Layer 4 (Attribution and Diagnostics): computes TreeSHAP for the prediction; cross-method agreement module flags high-uncertainty records. Layer 5 (Decision Support Outputs): generates all three application outputs.

_Architecture diagram guidance: Design Figure 7.1 as a horizontal flow diagram with five vertically stacked layer boxes. Each box has a label, a short function description, and the key technology (e.g., Layer 3: MambaAttention v10, SCCS #1, scikit-learn calibration). Arrows connect the layers left-to-right. A side branch shows the VIN ADAS enrichment connection from Layer 1 to the NHTSA vPIC database. Use the dissertation's blue colour scheme (#003366)._

### **_2.4 Section 7.4: Pipeline Foundation - The v10 Classification Engine (2-3 pages)_**

This section provides the explicit mapping between the v10 Jupyter notebook pipeline (06_Phase2_SAE_Classification_v10_ADAS) and the AVISTA system modules. It should include Table 7.2, which maps each notebook cell or phase to its corresponding AVISTA layer and module. The section explains the model selection decision: MambaAttention is deployed as the primary classification engine because it achieves SCCS #1 (0.5205), the highest combined score on operational penalty, calibration, and AA recall. The deployment model is frozen at the v10 checkpoint; incoming data is passed through the identical OHE and feature engineering pipeline that the training data used.

### **_2.5 Section 7.5: Interpretability Module (1-2 pages)_**

This section describes how SHAP attribution runs in the deployed tool. The key implementation note is that TreeSHAP is applied to the XGBoost model (not MambaAttention) for the SHAP attribution module, because XGBoost TreeSHAP provides exact, fast Shapley values with no approximation. The prediction itself comes from MambaAttention; the explanation comes from XGBoost. This dual-model approach is an acknowledged design tradeoff: the best-performing model (MambaAttention) provides the prediction, while the most interpretable model (XGBoost TreeSHAP) provides the attribution. This tradeoff should be explicitly stated and justified.

_Important academic honesty point: explicitly state in the text that the SHAP explanation refers to the XGBoost model's feature attribution, not the MambaAttention model's internal reasoning. This is a known limitation of the dual-model approach and should be presented as such, with the justification that XGBoost TreeSHAP is the gold-standard attribution method and XGBoost achieves competitive accuracy (SCCS #7, 0.2471 but Macro-F1 = 0.611 with reasonable calibration)._

### **_2.6 Section 7.6: Decision Support Application Modules (3-4 pages)_**

This is the core substantive section of the chapter. It describes each of the three application modules in detail, with one subsection each.

7.6.1 ODD Boundary Evaluator: inputs a crash record, returns P̂(AA|x) with calibrated uncertainty. Flagging logic: P̂(AA|x) > 0.50 = 'ODD-consistent AA record'; 0.25-0.50 = 'ambiguous zone'; < 0.25 = 'non-AA'. Include the SHAP waterfall for the individual record. Explain how this supports NHTSA's safety case review process.

7.6.2 Road Segment Risk Screener: aggregates individual predictions across all crash records in a TxDOT Atlas road segment. Outputs E\[P̂(AA|x)\] by weather×road-type combination. The 2D heatmap (Screenshot 5) is the primary output. Connects to the OOT weather robustness findings in Chapter 4 (Tables 4.17-4.18).

7.6.3 Policy Analytics Dashboard: uses the four-way XAI consensus results (Chapter 5, Tables 5.3-5.5) to display population-level feature importance for policy staff. Export capability to PDF for briefing documents. Discusses the Pop_Group finding as a data collection policy recommendation.

### **_2.7 Section 7.7: User Interface Design (2-3 pages)_**

This section is where all six screenshots are included. Each screenshot should be presented with a figure number, a descriptive caption, and 1-2 paragraphs of text explaining what the screen shows, what decisions it supports, and any notable UX design choices. The captions should be written from an expert transportation safety perspective, not just describing the UI elements.

For a dissertation, the screenshots are evidence that the tool exists and is functional. They do not need to show a polished production-grade UI - they need to show that the analytical pipeline is operational and the three decision-support applications produce interpretable output. Even a Jupyter-based or Streamlit prototype interface is sufficient, provided the outputs are correct and complete.

_Practical advice: If the full web implementation is not ready when you submit the dissertation, you can still include illustrative screenshots of: (a) the Streamlit/Gradio prototype built on the v10 pipeline, (b) the SHAP waterfall outputs from the XAI notebooks, (c) mock wireframe screens with real data values filled in. The chapter should be transparent about the tool's current implementation status (see Section 7.10) - a research prototype is a legitimate dissertation contribution._

### **_2.8 Section 7.8: Illustrative Use Case (1-2 pages)_**

An end-to-end walkthrough of a single realistic scenario. Suggested: A new crash record is submitted from an urban intersection in Dallas (Pop_Group = Group 1, Speed Limit = 30 mph, Wthr_Cond = Clear, ACC = yes, PAEB = yes, Obj_Struck = pedestrian, Body_Class = sedan, Light_Cond = daylight). The tool: (1) ingests the record; (2) engineers features; (3) predicts AA with P̂(AA) = 0.73, P̂(AD) = 0.21, P̂(PA) = 0.06; (4) flags as 'ODD-consistent AA'; (5) generates SHAP waterfall showing Pop_Group (+0.82), Speed_Limit (+0.61), ACC (+0.31) as top positive drivers; (6) road segment screening shows the E\[P̂(AA)\] for this Dallas arterial segment is 0.64 (high). Analyst takes action: flags record for review; recommends V2I camera upgrade for the segment.

## **Part 3: Screenshot Guide - What to Capture and Why**

This section provides a detailed guide to the six recommended screenshots for Chapter 7. Each screenshot description includes what must be visible, what values to use from the dissertation's actual results, and why the screenshot is compelling from a dissertation reviewer's perspective.

**Table 4. Six Recommended Screenshots - Content Specification and Dissertation Value.**

| **#** | **Screen / Panel**                       | **What It Should Show**                                                                                                                                                                                                                                                                   | **Why It Is Compelling**                                                                                                                                                                                                      |
| ----- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | **Landing Dashboard - Texas Crash Map**  | Interactive map of Texas with crash point layer coloured by predicted SAE class (blue=AD, amber=PA, red=AA). Sidebar shows class distribution pie chart. Date range and county filter controls visible.                                                                                   | Immediately communicates spatial intelligence. Shows AA concentration in DFW/Austin/Houston urban cores. Visually proves the Pop_Group finding from Ch5.                                                                      |
| **2** | **Crash Record Input Form**              | Clean input form with 24 fields pre-grouped: ADAS flags (checkbox panel), roadway context (speed limit, road type, weather, lighting dropdowns), crash configuration (object struck, intersection type). Submit button prominent.                                                         | Shows the tool is practitioner-usable - a TxDOT analyst can enter a new record and get an instant prediction. Grounds the abstract pipeline in a real workflow.                                                               |
| **3** | **Classification Output Panel**          | Predicted SAE class shown as large badge (e.g., 'ADVANCED AUTOMATION - SAE L3-5'). Three horizontal probability bars: AA=73.2%, AD=21.1%, PA=5.7%. Calibration confidence indicator (ECE shown). Model provenance label (MambaAttention v10).                                             | Demonstrates calibrated probabilistic output, not just a label. The three bars make uncertainty visible - a key requirement for safety-critical decision support.                                                             |
| **4** | **SHAP Waterfall Explanation Panel**     | SHAP waterfall chart for the specific prediction: base value shown at left (−0.64 for AA), then positive bars for Pop_Group (urban, +0.82) and Speed Limit (30 mph, +0.61), negative bars for Road_Type (−0.12), final output f(x)=+0.71 at right. Feature values annotated.              | The most academically impactful screenshot - directly visualises the connection between Ch5 XAI findings and the tool output. Shows AVISTA is not a black box.                                                                |
| **5** | **ODD Risk Profile - Road Segment View** | Heatmap table: rows = Road Type (Interstate, Urban Arterial, Rural Highway, FM Road), columns = Weather (Clear, Rain, Fog, Snow). Cells coloured by E\[P̂(AA)\] - darker = higher AA probability. A selected cell shows the count and mean probability. TxDOT corridor reference optional. | Directly answers RQ4's ODD boundary evaluation application. A 2D risk matrix is immediately interpretable by transportation engineers. Shows how the tool aggregates individual predictions to infrastructure-level insights. |
| **6** | **Policy Analytics Dashboard**           | Bar chart of top-10 feature SHAP importance (global, from Ch5 analysis). Filter by SAE class (AA/PA/AD toggle). Export-to-PDF button. Summary statistics panel (N records processed, date range, class distribution). Optional: weather robustness chart from Ch4.                        | Shows the policy development application. The export button signals that this is designed for practitioner output, not just academic display. Brings the SCCS and XAI results into one executive-level view.                  |

### **_3.1 Screenshot Priorities_**

If only two screenshots are possible (e.g., the tool is still in early development), prioritise Screenshots 3 (Classification Output Panel) and 4 (SHAP Waterfall Explanation). These two together tell the complete story of RQ4: the model predicts the SAE class with calibrated probabilities, and the XAI layer explains why. Screenshots 1 (crash map) and 5 (ODD heatmap) are the next priority because they show the spatial intelligence and infrastructure-level aggregation that distinguishes AVISTA from a generic prediction API.

For Screenshots 3 and 4, use the following actual values from the dissertation's results: P̂(AA) = 0.73, P̂(AD) = 0.21, P̂(PA) = 0.06 for a Dallas urban record. SHAP: Pop_Group (urban, +0.82), Speed_Limit (30 mph, +0.61), ACC (present, +0.31), Road_Type (urban arterial, +0.18), base value AA = −0.6405 (from SHAP expected value in Chapter 5). These are synthetic but internally consistent values grounded in the dissertation's actual SHAP results.

### **_3.2 Screenshot Placeholder Representations_**

The following placeholders illustrate where screenshots will be inserted in Chapter 7. Replace each placeholder with the actual screenshot when the tool interface is available.

**\[SCREENSHOT 1: Texas Crash Map Dashboard\]**

_Full-width map with SAE class colour layer (blue=AD, amber=PA, red=AA), sidebar class distribution pie chart, and date/county filters_

_Figure 7.1 (Screenshot 1). AVISTA landing dashboard showing interactive Texas crash map with SAE automation level overlay. AA crashes (red points) concentrate in DFW Metroplex, Austin, and Houston urban cores, visually confirming the Pop_Group_ID spatial discrimination finding from Chapter 5._

**\[SCREENSHOT 2: Crash Record Input Form\]**

_24-field input form with ADAS checkbox panel, roadway context dropdowns, crash configuration fields, and Submit button_

_Figure 7.2 (Screenshot 2). AVISTA crash record input form. Fields correspond directly to the 24 engineered features from the v10 pipeline. ADAS flags are presented as a checkbox grid; roadway context and crash configuration fields use dropdown menus populated from CRIS codebook values._

**\[SCREENSHOT 3: Classification Output Panel\]**

_Large SAE class badge, three probability bars, calibration confidence, model provenance label_

_Figure 7.3 (Screenshot 3). AVISTA SAE classification output for an illustrative urban Dallas crash record. The calibrated MambaAttention model (SCCS #1, v10) assigns P̂(AA)=0.73, P̂(AD)=0.21, P̂(PA)=0.06. The 'ODD-Consistent AA' flag is triggered at P̂(AA)>0.50._

**\[SCREENSHOT 4: SHAP Waterfall Explanation\]**

_SHAP waterfall from base value (−0.6405) to output (+0.71), feature contributions annotated with feature values_

_Figure 7.4 (Screenshot 4). AVISTA SHAP waterfall explanation for the classified record. The base value of −0.6405 (AA class prior) is incrementally adjusted by individual feature contributions: Pop_Group=urban (+0.82), Speed_Limit=30 mph (+0.61), ACC=present (+0.31), producing a final prediction of f(x)=+0.71 in log-odds space (P̂(AA)=0.73). This panel operationalises the Chapter 5 XAI findings at the individual prediction level._

**\[SCREENSHOT 5: ODD Risk Profile Heatmap\]**

_Road Type × Weather 2D heatmap coloured by E\[P̂(AA)\]; cell selected showing count and mean probability_

_Figure 7.5 (Screenshot 5). AVISTA ODD risk profile heatmap for the Dallas Urban Arterial road segment. Rows = road type (interstate, urban arterial, rural highway, FM road); columns = weather condition (clear, rain, fog, snow). Cell colour intensity represents expected P̂(AA) - darker cells indicate higher Advanced Automation concentration. Urban arterial + clear weather shows the highest AA probability (0.64), consistent with the deployed ODD of L3-5 platforms in fair-weather urban environments._

**\[SCREENSHOT 6: Policy Analytics Dashboard\]**

_Feature importance bar chart, SAE class toggle (AA selected), export button, summary statistics panel_

_Figure 7.6 (Screenshot 6). AVISTA policy analytics dashboard showing global feature importance for the Advanced Automation class. Bars represent normalised XGBoost SHAP importance; the top four features (Pop_Group_ID, Crash_Speed_Limit, Wthr_Cond_ID, Obj_Struck_ID) match the four-way XAI consensus from Chapter 5. The dashboard is filterable by SAE class and exportable to PDF for regulatory briefing documents._

## **Part 4: Framing AVISTA as a Dissertation Contribution - Key Arguments**

### **_4.1 Can We Say AVISTA Is a Dissertation Contribution? Yes - and Here Is How._**

The answer is clearly yes, and the dissertation's abstract already makes this claim. The four contributions listed in the abstract include: (1) novel methodological framework; (2) comprehensive validation strategy; (3) policy and safety insights; (4) 'the implementation of AVISTA as a structured analytical decision-support framework.' Contribution (4) is the most directly linked to Chapter 7.

The key framing language to use in the chapter is 'research prototype' or 'proof-of-concept implementation.' This accurately describes a tool that has been developed to demonstrate that the pipeline is operationalisable, without overstating completion of a production-grade deployment. Engineering and computer science dissertations regularly claim tool development as a contribution at the prototype stage - the contribution is the design, architecture, and analytical capability of the tool, not the scalability of the production deployment.

_Suggested contribution claim language for Chapter 7: 'This chapter presents the AVISTA research prototype - a web-based analytical decision-support system that operationalises the SAE automation level classification engine and four-method XAI attribution pipeline from Chapters 4 and 5 into three structured decision-support applications. AVISTA constitutes the fourth contribution of this dissertation: the translation of empirical machine learning findings into a practitioner-accessible tool for transportation safety analysis.'_

### **_4.2 Distinguishing 'Conceptual Framework' (Chapter 5) from 'Tool Implementation' (Chapter 7)_**

Chapter 5, Section 5.7 describes the AVISTA framework conceptually - its architecture layers, its three application modules, and its design rationale. Chapter 7 goes further: it shows the system working. The distinction is important for a dissertation reviewer who might ask 'what was actually built?' Chapter 5 answers 'here is what AVISTA is designed to do.' Chapter 7 answers 'here is AVISTA doing it' - with screenshots, use cases, and validation provenance.

This two-level treatment (conceptual in Ch5, operational in Ch7) is standard in dissertations that include both an analytical contribution and a tool contribution. The analytical contribution (Chapters 4-5) demonstrates feasibility and produces validated results. The tool contribution (Chapter 7) demonstrates operability and produces practical outputs. Neither chapter is redundant with the other.

### **_4.3 Connection to All Four Research Questions_**

Although Chapter 7 is primarily motivated by RQ4, a well-written AVISTA chapter implicitly validates all four RQs by demonstrating that the tool's outputs are grounded in the answers to each. The classification output panel (Screenshot 3) demonstrates RQ1 (accurate classification is possible). The SHAP waterfall (Screenshot 4) operationalises RQ2 (feature attribution in practice). The weather condition filtering in the ODD risk screener (Screenshot 5) reflects RQ3 (infrastructure robustness affects risk profiles). The policy dashboard (Screenshot 6) directly answers RQ4. This cross-RQ integration is a compelling argument for the dissertation committee: AVISTA is not just a tool chapter - it is the dissertation's synthesis chapter.

## **Part 5: Writing Guidance and Style Notes for Chapter 7**

### **_5.1 Tone and Register_**

Chapter 7 can be slightly less dense with statistical tables than Chapters 4 and 5 - it is more descriptive and system-oriented. The appropriate tone is the same 'transportation safety expert analyst' voice used in Chapters 4 and 5, but applied to system design and usability rather than statistical results. Phrases like 'the tool enables a TxDOT analyst to...', 'this output supports ODD regulatory review by...', and 'the policy dashboard translates the Chapter 5 XAI findings into an exportable decision brief' keep the reader grounded in practical transportation safety relevance.

### **_5.2 What Tables Chapter 7 Needs_**

Table 7.1: AVISTA Contribution Summary (mapping each contribution type to evidence in the dissertation). Table 7.2: v10 Pipeline to AVISTA Module Mapping (notebook phase → system layer). Table 7.3: Decision Support Application Module Summary (Application name / Input / Processing / Output / RQ addressed). Table 7.4: Use Case Walk-Through Steps (numbered steps with tool action and output for each). Table 7.5: RQ4 Answer Summary (replaces the brief Table 5.8 entry from Chapter 5 with a full RQ4 answer grounded in tool capabilities).

### **_5.3 What Figures Chapter 7 Needs_**

Figure 7.1: AVISTA System Architecture Diagram (the five-layer diagram - this is the most important figure and should be drawn/designed properly, not a placeholder). Figures 7.2-7.7: The six screenshots (Figures 7.2 = Screenshot 1 through Figure 7.7 = Screenshot 6). Figure 7.8: Optional - the end-to-end use case flow diagram showing the Dallas record being processed through all five layers.

### **_5.4 Estimated Chapter Length_**

When written in full with all tables and figures: approximately 25-35 pages. This is appropriate for a 'system description + use case' chapter. It is shorter than Chapters 4 and 5 (which are 50-80 pages each) but longer than a typical appendix, justifying its status as a chapter rather than supplemental material.

**- END OF AVISTA CHAPTER PLANNING DOCUMENT -**

_Shriyank Somvanshi - PhD Dissertation, Texas State University - June 2026_