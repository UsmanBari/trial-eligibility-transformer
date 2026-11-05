"""trialmatch: criterion-level clinical-trial eligibility with a decoder Transformer."""

INCLUSION_LABELS = ["included", "not included", "not enough information", "not applicable"]
EXCLUSION_LABELS = ["excluded", "not excluded", "not enough information", "not applicable"]
ALL_LABELS = ["included", "not included", "excluded", "not excluded",
              "not enough information", "not applicable"]
INVALID_LABEL = "invalid"   # model output we could not parse or that is not allowed for the criterion type


def allowed_labels(criterion_type: str):
    t = criterion_type.strip().lower()
    if t == "inclusion":
        return INCLUSION_LABELS
    if t == "exclusion":
        return EXCLUSION_LABELS
    raise ValueError(f"unknown criterion_type: {criterion_type!r}")
