# Error analysis: `ft_v2_test_greedy` (test split, 206 criteria)

| Category | Count |
|---|---|
| Inclusion decision reversed (included <-> not included) | 0 |
| Exclusion decision reversed (excluded <-> not excluded) | 0 |
| Model decided although the expert said not enough information (absent evidence treated as proof) | 30 |
| Model said not enough information although the note does decide it | 20 |
| Confusion involving not applicable | 14 |
| Unparseable output or label not allowed for the criterion type | 0 |
| Correct label, but no sentence cited although the expert cited some | 9 |
| Correct label, cited sentences do not overlap the expert's | 4 |
| Cited sentence IDs that do not exist in the note | 0 |

## Model decided although the expert said not enough information (absent evidence treated as proof)

### Case 1: annotation 174 (patient sigir-201410, NCT02097186, exclusion)
**Criterion:** Patients with previous axillary surgery  
**Expert:** `not enough information`, evidence []  
**Model:** `not excluded`, evidence []  
**Raw output:** `{"label": "not excluded", "evidence": []}`

Expert evidence sentences:
  - (none)
Model evidence sentences:
  - (none)

### Case 2: annotation 193 (patient sigir-201411, NCT01526382, exclusion)
**Criterion:** Patients were moribund.  
**Expert:** `not enough information`, evidence [0, 1, 2, 3, 4]  
**Model:** `not excluded`, evidence []  
**Raw output:** `{"label": "not excluded", "evidence": []}`

Expert evidence sentences:
  - [0] A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.
  - [1] She denies trauma.
  - [2] On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.
  - [3] Her body temperature is normal and her blood pressure is 80/60.
  - [4] Her right arm has no discoloration or movement limitation.
Model evidence sentences:
  - (none)


## Model said not enough information although the note does decide it

### Case 3: annotation 162 (patient sigir-201410, NCT02097186, inclusion)
**Criterion:** Patients undergoing open abdominal aortic aneurysm repair  
**Expert:** `not included`, evidence []  
**Model:** `not enough information`, evidence [0]  
**Raw output:** `{"label": "not enough information", "evidence": [0]}`

Expert evidence sentences:
  - (none)
Model evidence sentences:
  - [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.

### Case 4: annotation 163 (patient sigir-201410, NCT02097186, inclusion)
**Criterion:** Patients undergoing endovascular abdominal aneurysm repair  
**Expert:** `not included`, evidence []  
**Model:** `not enough information`, evidence [0, 2]  
**Raw output:** `{"label": "not enough information", "evidence": [0, 2]}`

Expert evidence sentences:
  - (none)
Model evidence sentences:
  - [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
  - [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.


## Confusion involving not applicable

### Case 5: annotation 322 (patient sigir-201417, NCT01745731, exclusion)
**Criterion:** Pregnant women or women of childbearing age who do not have adequate contraception.  
**Expert:** `not excluded`, evidence [0]  
**Model:** `not applicable`, evidence []  
**Raw output:** `{"label": "not applicable", "evidence": []}`

Expert evidence sentences:
  - [0] A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.
Model evidence sentences:
  - (none)

### Case 6: annotation 407 (patient sigir-201421, NCT00997100, exclusion)
**Criterion:** Female subject of child-bearing potential who is not using a medically accepted safe method of contraception. All female subjects of child-bearing potential must have a negative urine pregnancy test at the Screening and Baseline Visits. As interaction studies between ABR-215757 and oral contraceptives have not yet been performed, women using the contraceptive pill must also use a complementary contraceptive device, i.e. barrier method, during the treatment period and for at least 1 month thereafter.  
**Expert:** `not enough information`, evidence []  
**Model:** `not applicable`, evidence []  
**Raw output:** `{"label": "not applicable", "evidence": []}`

Expert evidence sentences:
  - (none)
Model evidence sentences:
  - (none)


## Correct label, but no sentence cited although the expert cited some

### Case 7: annotation 176 (patient sigir-201410, NCT02264964, inclusion)
**Criterion:** No medical history of central vena catheterization.  
**Expert:** `included`, evidence [0]  
**Model:** `included`, evidence []  
**Raw output:** `{"label": "included", "evidence": []}`

Expert evidence sentences:
  - [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
Model evidence sentences:
  - (none)

### Case 8: annotation 179 (patient sigir-201410, NCT02264964, exclusion)
**Criterion:** Had been performed central venous puncture or catheterization before.  
**Expert:** `not excluded`, evidence [0]  
**Model:** `not excluded`, evidence []  
**Raw output:** `{"label": "not excluded", "evidence": []}`

Expert evidence sentences:
  - [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
Model evidence sentences:
  - (none)


## Correct label, cited sentences do not overlap the expert's

### Case 9: annotation 326 (patient sigir-201417, NCT02556359, exclusion)
**Criterion:** Refusal to consent.  
**Expert:** `not excluded`, evidence [5]  
**Model:** `not excluded`, evidence [4]  
**Raw output:** `{"label": "not excluded", "evidence": [4]}`

Expert evidence sentences:
  - [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
Model evidence sentences:
  - [4] After surgery, the patient is taken to the ICU.

### Case 10: annotation 381 (patient sigir-201421, NCT00997100, inclusion)
**Criterion:** Age > 18 years at the time of signing the informed consent form  
**Expert:** `included`, evidence [0]  
**Model:** `included`, evidence [4]  
**Raw output:** `{"label": "included", "evidence": [4]}`

Expert evidence sentences:
  - [0] A 21-year-old female is evaluated for progressive arthralgias and malaise.
Model evidence sentences:
  - [4] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
