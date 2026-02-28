# Student Performance & Churn Analysis - Data Preprocessing

## 1. Project Overview
The goal of this phase was to transform a raw, Hungarian-language university student database into a standardized, clean dataset. This prepared data serves as the foundation for Exploratory Data Analysis (EDA) and future predictive modeling for student churn.

## 2. Challenges & Problem Statement
Before processing, the dataset presented several obstacles:
* **Language Barriers:** Headers and categorical values (e.g., funding types, enrollment statuses) were in Hungarian, complicating the use of standard data science libraries (Pandas, Seaborn).
* **Significant Data Gaps:** Roughly **16.5%** of records lacked academic performance data (GPA, credits).
* **Non-standard Date Formats:** Birth dates were stored as inconsistent strings (e.g., `dec.91`), preventing numerical age calculations.
* **Integrity Issues:** Missing ZIP codes and inconsistent data types required resolution for spatial and temporal analysis.

## 3. Applied Solutions & Logic

### 3.1 Standardization & Translation
* **Header Mapping:** All columns were renamed to standardized English technical terms (e.g., *Modulkód* $\rightarrow$ `major_code`).
* **Categorical Mapping:** Dictionary-based mapping was used to translate 21 unique student statuses and 26 funding categories. We used the `.replace()` method to ensure unmapped values remained visible rather than being lost to `NaN`.

### 3.1 Handling Missing Values (Data Integrity)
Following a consistent cleaning strategy:
* **Academic Performance (GPA & Credits):** Missing values (the ~16.5% identified) were filled with **0.0**. 
  * *Rationale:* In this context, a missing grade often represents inactivity or failure, which is a crucial signal for churn prediction models.
* **Critical Records:** Rows missing unique identifiers (`student_id`) or core enrollment data were **dropped** to maintain dataset reliability.
* **ZIP Codes:** Since only a negligible portion of rows was missing location data, these were dropped to ensure smooth geographic grouping.

### 3.3 Geographic & Regional Enrichment
* **ZIP-to-County Mapping:** The postal codes were too granular for high-level analysis. 
    * **Logic:** Using a custom mapping based on the first two digits of the `zip_code`, students were categorized into 20 regions (19 Hungarian counties + Budapest).
    * **Result:** Created the `county` column, enabling spatial churn rate analysis (e.g., identifying if students from remote counties are more likely to drop out).

### 3.4 Academic & Institutional Mapping
* **Deciphering Major Codes:** Raw data contained cryptic codes (e.g., `2BNPSZV`, `7BNKOMM`).
    * **Mapping:** Based on university faculty logic (BCE/MATE), 26 unique codes were mapped to full English degree names (e.g., *Finance and Accounting BSc*).
    * **Institutional Separation:** Identified the specific university (`university`) for each record.
    * **Broad Field Grouping:** Grouped specific majors into 5-6 broad disciplines (e.g., *IT, Economics, Social Sciences*) to increase statistical significance during EDA.



### 3.5 Advanced Feature Engineering (The "Analytical" Layer)
To enhance predictive power, we derived new features:
* **Academic Efficiency Ratio:** Created the `academic_efficiency` column ($Credits\ Done\ Cum\ /\ Credits\ Taken\ Cum$). 
    * *Insight:* This represents how successfully a student completes what they attempt, often a better indicator than GPA alone.
* **Age at Enrollment:** Calculated by subtracting the birth year from the enrollment year.
    * *Rationale:* Helps differentiate between "fresh" high-school graduates and mature students returning from the workforce.
* **Target Variable Definition (`is_churned`):** Defined the core outcome.
    * **Logic:** Students with statuses such as 'Deleted', 'Left without notice', or 'Failed final exam' were flagged as **1 (Churned)**, while active or graduated students were flagged as **0**.



## 4. Final Data Schema (Analytical Base Table)
The final processed dataset contains **29 columns** organized into logical groups:

| Group | Columns |
| :--- | :--- |
| **Identifiers & Dates** | `student_id`, `semester_id`, `term_number`, `enrollment_date`, `enrollment_year` |
| **Demographics** | `gender`, `birth_year`, `age_at_enrollment` |
| **Geography** | `zip_code`, `county` |
| **Academic Background** | `university`, `major_code`, `major_name_en`, `major_field_broad`, `degree_level`, `study_mode` |
| **Financials & Status** | `funding_type`, `funding_category`, `semester_active_status`, `status_at_record` |
| **Performance Metrics** | `total_active_terms`, `credits_taken_term`, `credits_taken_cum`, `credits_done_term`, `credits_done_cum`, `gpa_term`, `gpa_cum`, `academic_efficiency` |
| **Target Variable** | **`is_churned`** |

## 5. Summary of Results
* **Cleanliness:** 0% missing values in critical columns after cleaning.
* **Standardization:** 100% English-compliant naming and values.
* **Enrichment:** High-value features added for better churn prediction (Efficiency, Age, County).
* **Ready for EDA:** The dataset is fully optimized for the next phase of exploratory visualization.