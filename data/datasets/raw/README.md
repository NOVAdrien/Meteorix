# Image Dataset for Automatic Meteor Fall Detection

## Dataset origin

This README documents the dataset imported from Mendeley Data:

- **Dataset title:** Image dataset for the creation of an automatic system for meteor fall detection
- **Mendeley Data DOI:** [10.17632/4k98n84d9g.1](https://doi.org/10.17632/4k98n84d9g.1)
- **Dataset URL:** https://data.mendeley.com/datasets/4k98n84d9g/1
- **Version:** 1
- **Published:** 27 June 2023
- **Contributors:** Andre Gradvohl; Victor Yukio Shirasuna
- **Institution:** Universidade Estadual de Campinas
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Categories:** Astronomy; Meteor

## Recommended citation

Gradvohl, A.; Shirasuna, V. Y. (2023). *Image dataset for the creation of an automatic system for meteor fall detection*. Mendeley Data, V1. https://doi.org/10.17632/4k98n84d9g.1

When reusing this dataset, cite the original Mendeley Data record and keep attribution to the original contributors and image sources.

## Description

This dataset contains sky images showing either the occurrence or non-occurrence of falling meteors. It was created to support the development of an automatic meteor fall detection system using deep learning.

The dataset contains **7,000 JPEG images**:

- **3,850 images** with meteors, representing approximately **55%** of the dataset
- **3,150 images** without meteors, representing approximately **45%** of the dataset

The images were captured by different instruments between **2014 and 2023**.

## Original image sources

The primary data sources listed by the dataset authors are:

- **BRAMON** — Brazilian Meteor Observation Network: http://www.bramonmeteor.org
- **UKMON** — UK Meteor Network: https://ukmeteornetwork.co.uk
- **BOAM** — Base des Observateurs Amateurs de Météores: http://boam.fr

## Folder structure

The original dataset is organized into two folder levels.

### Level 1

```text
RawImages/
CroppedImages/
```

- `RawImages/`: images including captions as stored in the original repositories
- `CroppedImages/`: images without captions; a 24-pixel band was cropped from the lower part of the image

### Level 2

Each of the two top-level folders contains:

```text
meteor/
non-meteors/
```

- `meteor/`: images containing meteors
- `non-meteors/`: images without meteor occurrences

Expected structure:

```text
RawImages/
├── meteor/
└── non-meteors/

CroppedImages/
├── meteor/
└── non-meteors/
```

## File naming convention

### Meteor images

Meteor image filenames follow this pattern:

```text
<source>_<date>_<id>.jpg
```

Where:

- `<source>` is one of `bramon`, `ukmon`, or `boam`
- `<date>` is the capture date and time in the format `yyyymmdd_hhnnss`
- `<id>` is a source-specific identifier used to avoid date-time conflicts:
  - BRAMON: radiant identifier
  - UKMON: station identifier
  - BOAM: station identifier

### Non-meteor images

Non-meteor image filenames follow this pattern:

```text
<source>_<date>_nonmeteor.jpg
```

This pattern avoids duplicate names and explicitly identifies images without meteors.

## Suggested labels for computer vision workflows

For classification workflows, the dataset can be treated as a binary classification dataset with the following labels:

```text
meteor
non-meteor
```

For Roboflow classification projects, a practical mapping is:

| Original folder | Suggested class label |
|---|---|
| `meteor/` | `meteor` |
| `non-meteors/` | `non-meteor` |

## Suggested Roboflow metadata

When importing this dataset into Roboflow, document the source using project metadata, project description, tags, and/or batch names.

Suggested project description:

```text
Dataset imported from Mendeley Data: "Image dataset for the creation of an automatic system for meteor fall detection", Version 1, DOI: 10.17632/4k98n84d9g.1. Original contributors: Andre Gradvohl and Victor Yukio Shirasuna. Licensed under CC BY 4.0. Original sources include BRAMON, UKMON, and BOAM meteor observation repositories.
```

Suggested tags:

```text
source:mendeley-data
source-doi:10.17632/4k98n84d9g.1
license:cc-by-4.0
domain:astronomy
task:meteor-detection
```

Suggested batch names:

```text
mendeley_meteor_dataset_v1_raw
mendeley_meteor_dataset_v1_cropped
```

## License and attribution notes

The Mendeley Data record lists the dataset license as **CC BY 4.0**. Under CC BY 4.0, reuse is allowed provided that appropriate attribution is given, a link to the license is provided, and changes are indicated when applicable.

Suggested attribution text:

```text
This dataset is based on "Image dataset for the creation of an automatic system for meteor fall detection" by Andre Gradvohl and Victor Yukio Shirasuna, published on Mendeley Data, Version 1, DOI: 10.17632/4k98n84d9g.1, licensed under CC BY 4.0.
```

The full license text is available in:
- `LICENSE-CC-BY-4.0.txt`

## Notes for model training

- Use `CroppedImages/` if captions or embedded text in the lower part of the image could introduce bias or shortcuts during training.
- Keep train, validation, and test splits independent across closely related captures where possible.
- Preserve source metadata when possible, because images originate from multiple meteor observation networks and instruments.
- Review class balance before training: the original dataset contains about 55% meteor images and 45% non-meteor images.
