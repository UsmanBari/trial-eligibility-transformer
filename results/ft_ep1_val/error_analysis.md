# Error analysis: `ft_ep1_val` (val split, 206 criteria)

| Category | Count |
|---|---|
| Inclusion decision reversed (included <-> not included) | 2 |
| Exclusion decision reversed (excluded <-> not excluded) | 1 |
| Model decided although the expert said not enough information (absent evidence treated as proof) | 31 |
| Model said not enough information although the note does decide it | 10 |
| Confusion involving not applicable | 12 |
| Unparseable output or label not allowed for the criterion type | 0 |
| Correct label, but no sentence cited although the expert cited some | 0 |
| Correct label, cited sentences do not overlap the expert's | 16 |
| Cited sentence IDs that do not exist in the note | 3 |

## Inclusion decision reversed (included <-> not included)

### Case 1: annotation 934 (patient sigir-201522, NCT02534727, inclusion)
**Criterion:** Taking anti-tuberculosis medicines (or NTM meds at NIH CC) during the time sputa are provided  
**Expert:** `not included`, evidence []  
**Model:** `included`, evidence [0]  
**Raw output:** `{"label": "included", "evidence": [0]}`

Expert evidence sentences:
  - (none)
Model evidence sentences:
  - [0] A 65-year-old male with a history of tuberculosis has started to complain of productive cough with tinges of blood.

### Case 2: annotation 935 (patient sigir-201522, NCT02534727, inclusion)
**Criterion:** Thought likely to be Mycobacterium culture positive (including NTM infected for the NIH CC) by enrolling physician  
**Expert:** `not included`, evidence [3]  
**Model:** `included`, evidence [3]  
**Raw output:** `{"label": "included", "evidence": [3]}`

Expert evidence sentences:
  - [3] Culture of the sputum revealed an organism with septated, low-angle branching hyphae that had straight, parallel walls.
Model evidence sentences:
  - [3] Culture of the sputum revealed an organism with septated, low-angle branching hyphae that had straight, parallel walls.


## Exclusion decision reversed (excluded <-> not excluded)

### Case 3: annotation 263 (patient sigir-201415, NCT00180739, exclusion)
**Criterion:** 13. Evidence of uterine pathology other than leiomyoma  
**Expert:** `excluded`, evidence [5]  
**Model:** `not excluded`, evidence [4]  
**Raw output:** `{"label": "not excluded", "evidence": [4]}`

Expert evidence sentences:
  - [5] The differential diagnosis includes vesicular mole vs fibroid degeneration.
Model evidence sentences:
  - [4] The ultrasound reports enlarged uterus (12 cm x 9 cm x 7 cms) with multiple cystic areas in the interior.


## Model decided although the expert said not enough information (absent evidence treated as proof)

### Case 4: annotation 8 (patient sigir-20141, NCT01397994, exclusion)
**Criterion:** Unstable angina  
**Expert:** `not enough information`, evidence [0, 1, 2]  
**Model:** `not excluded`, evidence [0]  
**Raw output:** `{"label": "not excluded", "evidence": [0]}`

Expert evidence sentences:
  - [0] A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.
  - [1] The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration.
  - [2] The latest episode of pain ended half an hour prior to her arrival.
Model evidence sentences:
  - [0] A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.

### Case 5: annotation 15 (patient sigir-20141, NCT01397994, exclusion)
**Criterion:** Pregnant and nursing women  
**Expert:** `not enough information`, evidence []  
**Model:** `not excluded`, evidence [0]  
**Raw output:** `{"label": "not excluded", "evidence": [0]}`

Expert evidence sentences:
  - (none)
Model evidence sentences:
  - [0] A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.


## Model said not enough information although the note does decide it

### Case 6: annotation 31 (patient sigir-20142, NCT02618655, inclusion)
**Criterion:** patients have fever more than one week  
**Expert:** `not included`, evidence [0]  
**Model:** `not enough information`, evidence [0]  
**Raw output:** `{"label": "not enough information", "evidence": [0]}`

Expert evidence sentences:
  - [0] An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days.
Model evidence sentences:
  - [0] An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days.

### Case 7: annotation 32 (patient sigir-20142, NCT02618655, inclusion)
**Criterion:** temperature is higher than 38℃ Celsius degree  
**Expert:** `included`, evidence [0]  
**Model:** `not enough information`, evidence [0]  
**Raw output:** `{"label": "not enough information", "evidence": [0]}`

Expert evidence sentences:
  - [0] An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days.
Model evidence sentences:
  - [0] An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days.


## Confusion involving not applicable

### Case 8: annotation 24 (patient sigir-20141, NCT00149227, exclusion)
**Criterion:** Patients with IHD within 6 months after percutaneous coronary intervention(PCI), and who are stable but are going to implement PCI or coronary artery bypass grafting(CABG)  
**Expert:** `not applicable`, evidence []  
**Model:** `not excluded`, evidence [0, 3, 4]  
**Raw output:** `{"label": "not excluded", "evidence": [0, 3, 4]}`

Expert evidence sentences:
  - (none)
Model evidence sentences:
  - [0] A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.
  - [3] She is known to have hypertension and obesity.
  - [4] She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease.

### Case 9: annotation 33 (patient sigir-20142, NCT02618655, inclusion)
**Criterion:** full of physical examination and laboratory examination have been carried out after one week，but still cannot make a definite diagnosis  
**Expert:** `not applicable`, evidence []  
**Model:** `not enough information`, evidence [4, 5]  
**Raw output:** `{"label": "not enough information", "evidence": [4, 5]}`

Expert evidence sentences:
  - (none)
Model evidence sentences:
  - [4] On examination he is in respiratory distress and has bronchial respiratory sounds on the left.
  - [5] A chest x-ray shows bilateral lung infiltrates.


## Correct label, cited sentences do not overlap the expert's

### Case 10: annotation 17 (patient sigir-20141, NCT01397994, exclusion)
**Criterion:** On calcium channel blockers  
**Expert:** `not excluded`, evidence [5]  
**Model:** `not excluded`, evidence [7]  
**Raw output:** `{"label": "not excluded", "evidence": [7]}`

Expert evidence sentences:
  - [5] She currently takes no medications.
Model evidence sentences:
  - [7] The EKG shows nonspecific changes.

### Case 11: annotation 20 (patient sigir-20141, NCT01397994, exclusion)
**Criterion:** Geographical inaccessibility for treatment or follow-up evaluations  
**Expert:** `not excluded`, evidence [8]  
**Model:** `not excluded`, evidence [0, 1, 2, 3, 4, 5, 6, 7]  
**Raw output:** `{"label": "not excluded", "evidence": [0, 1, 2, 3, 4, 5, 6, 7]}`

Expert evidence sentences:
  - [8] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
Model evidence sentences:
  - [0] A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.
  - [1] The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration.
  - [2] The latest episode of pain ended half an hour prior to her arrival.
  - [3] She is known to have hypertension and obesity.
  - [4] She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease.
  - [5] She currently takes no medications.
  - [6] Physical examination is normal.
  - [7] The EKG shows nonspecific changes.


## Cited sentence IDs that do not exist in the note

### Case 12: annotation 503 (patient sigir-201427, NCT01187901, inclusion)
**Criterion:** Adequate bone marrow function as show by: normal leukocyte count, platelet count ≥ 120 x 109/L, Hgb > 12 g/dL  
**Expert:** `not enough information`, evidence []  
**Model:** `not enough information`, evidence []  
**Raw output:** `{"label": "not enough information", "evidence": [6]}`

Expert evidence sentences:
  - (none)
Model evidence sentences:
  - (none)

### Case 13: annotation 504 (patient sigir-201427, NCT01187901, inclusion)
**Criterion:** Adequate liver function as shown by: normal serum bilirubin(≤ 1.5 Upper Limit Normal {ULN}) and serum transaminases (≤ 2.0 ULN)  
**Expert:** `not enough information`, evidence []  
**Model:** `not enough information`, evidence []  
**Raw output:** `{"label": "not enough information", "evidence": [6]}`

Expert evidence sentences:
  - (none)
Model evidence sentences:
  - (none)
