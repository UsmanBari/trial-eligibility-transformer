# Error analysis: `ft_best_test` (test split, 206 criteria)

| Category | Count |
|---|---|
| Inclusion decision reversed (included <-> not included) | 0 |
| Exclusion decision reversed (excluded <-> not excluded) | 0 |
| Model decided although the expert said not enough information (absent evidence treated as proof) | 30 |
| Model said not enough information although the note does decide it | 14 |
| Confusion involving not applicable | 15 |
| Unparseable output or label not allowed for the criterion type | 0 |
| Correct label, but no sentence cited although the expert cited some | 0 |
| Correct label, cited sentences do not overlap the expert's | 4 |
| Cited sentence IDs that do not exist in the note | 0 |

## Model decided although the expert said not enough information (absent evidence treated as proof)

### Case 1: annotation 169 (patient sigir-201410, NCT02097186, exclusion)
**Criterion:** Patients with an estimated pre-operative glomerular filtration rate < 30mls/min/1.73m2  
**Expert:** `not enough information`, evidence []  
**Model:** `not excluded`, evidence [0]  
**Raw output:** `{"label": "not excluded", "evidence": [0]}`

Expert evidence sentences:
  - (none)
Model evidence sentences:
  - [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.

### Case 2: annotation 174 (patient sigir-201410, NCT02097186, exclusion)
**Criterion:** Patients with previous axillary surgery  
**Expert:** `not enough information`, evidence []  
**Model:** `not excluded`, evidence [0]  
**Raw output:** `{"label": "not excluded", "evidence": [0]}`

Expert evidence sentences:
  - (none)
Model evidence sentences:
  - [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.


## Model said not enough information although the note does decide it

### Case 3: annotation 161 (patient sigir-201410, NCT02097186, inclusion)
**Criterion:** Patients undergoing elective carotid endarterectomy  
**Expert:** `not included`, evidence []  
**Model:** `not enough information`, evidence [0]  
**Raw output:** `{"label": "not enough information", "evidence": [0]}`

Expert evidence sentences:
  - (none)
Model evidence sentences:
  - [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.

### Case 4: annotation 162 (patient sigir-201410, NCT02097186, inclusion)
**Criterion:** Patients undergoing open abdominal aortic aneurysm repair  
**Expert:** `not included`, evidence []  
**Model:** `not enough information`, evidence [0]  
**Raw output:** `{"label": "not enough information", "evidence": [0]}`

Expert evidence sentences:
  - (none)
Model evidence sentences:
  - [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.


## Confusion involving not applicable

### Case 5: annotation 165 (patient sigir-201410, NCT02097186, exclusion)
**Criterion:** Pregnancy  
**Expert:** `not applicable`, evidence []  
**Model:** `not excluded`, evidence [0]  
**Raw output:** `{"label": "not excluded", "evidence": [0]}`

Expert evidence sentences:
  - (none)
Model evidence sentences:
  - [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.

### Case 6: annotation 422 (patient sigir-201422, NCT00236912, inclusion)
**Criterion:** If female, using birth control  
**Expert:** `not applicable`, evidence [3]  
**Model:** `not enough information`, evidence [3]  
**Raw output:** `{"label": "not enough information", "evidence": [3]}`

Expert evidence sentences:
  - [3] She has had no sexual partners and her menses are regular.
Model evidence sentences:
  - [3] She has had no sexual partners and her menses are regular.


## Correct label, cited sentences do not overlap the expert's

### Case 7: annotation 197 (patient sigir-201411, NCT01723137, exclusion)
**Criterion:** Decreased level of consciousness  
**Expert:** `not excluded`, evidence [0, 5]  
**Model:** `not excluded`, evidence [2]  
**Raw output:** `{"label": "not excluded", "evidence": [2]}`

Expert evidence sentences:
  - [0] A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.
  - [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
Model evidence sentences:
  - [2] On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.

### Case 8: annotation 319 (patient sigir-201417, NCT01745731, exclusion)
**Criterion:** Difficulty in monitoring.  
**Expert:** `not excluded`, evidence [5]  
**Model:** `not excluded`, evidence [0, 1, 2, 3, 4]  
**Raw output:** `{"label": "not excluded", "evidence": [0, 1, 2, 3, 4]}`

Expert evidence sentences:
  - [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
Model evidence sentences:
  - [0] A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.
  - [1] The physical examination is remarkable for tenderness and positive Murphy sign.
  - [2] Abdominal ultrasound shows hepatomegaly and abundant free intraperitoneal fluid.
  - [3] Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained.
  - [4] After surgery, the patient is taken to the ICU.
