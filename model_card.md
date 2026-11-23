# Model card: Fingerprint Study Lab baseline

## Status

Experimental educational baseline. Not clinically validated, not approved for medical use, and not suitable for use with real patient or donor decisions.

## Intended use

Teaching dataset hygiene, participant-aware evaluation, image classification, error analysis, and model documentation using a consented, de-identified research dataset.

## Prohibited use

- Inferring, reporting, or confirming a person's blood group
- Transfusion, donation, pregnancy, emergency, or diagnostic decisions
- Uploading real fingerprints to the public demonstration site
- Representing a model score as a laboratory result

## Training design

The baseline is a ResNet-18 image classifier trained from scratch. Images are resized to 224×224 and augmented only in the training split. Split assignment is at participant level, not image level. A saved evaluation report includes macro-F1, a confusion matrix, and a confidence distribution.

## Required evidence before any broader research claim

1. Provenance, consent, and label-verification documentation for every record.
2. Participant-disjoint train/validation/test splits.
3. An external dataset from a distinct collection setting.
4. Class balance, demographic representation, and acquisition-device analysis.
5. Calibration, abstention behaviour, and error analysis by subgroup.
6. Independent expert review and an appropriate ethics/IRB process.

Even satisfying these steps would not make the system a blood-typing test. Valid blood grouping uses a blood sample and validated laboratory methods.
