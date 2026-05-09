# Meteor / Non-Meteor Image Dataset

## Overview

This dataset contains cropped images classified into two categories:

- **Meteor images**
- **Non-meteor images**

The original dataset contains **7,000 images** in total.

| Class | Number of Images |
|---|---:|
| Images with meteors | 3,850 |
| Images without meteors | 3,150 |
| **Total** | **7,000** |

## Dataset Balancing

The objective is to build a balanced working dataset with the following class distribution:

- **70% images with meteors**
- **30% images without meteors**

Since all **3,850 meteor images** are kept, the final dataset size is calculated as follows:

- 3,850 meteor images represent 70% of the final dataset
- Final dataset size: **5,500 images**
- Required non-meteor images: **1,650 images**

Therefore:

| Class | Images Kept | Images Discarded |
|---|---:|---:|
| Images with meteors | 3,850 | 0 |
| Images without meteors | 1,650 | 1,500 |
| **Total** | **5,500** | **1,500** |

## Dataset Split

The resulting dataset of **5,500 images** is split into three subsets:

- `training`
- `test`
- `valid`

The split follows an **80% / 10% / 10%** ratio:

| Subset | Percentage | Number of Images |
|---|---:|---:|
| `training` | 80% | 4,400 |
| `test` | 10% | 550 |
| `valid` | 10% | 550 |
| **Total** | **100%** | **5,500** |

## Source Folder

Images are selected from the following folder:

```text
CroppedImages/
```

The dataset is organized using the following class folders:

```text
CroppedImages/
├── meteor/
└── nonmeteor/
```

## Important Note

In the `CroppedImages/nonmeteor` folder, three images are similar. Because of this, only one of these three similar images have been annotated with Roboflow. In the `CroppedImages/meteor` folder too, two images are similar, so only one of these two similar images has been annotated with Roboflow.

As a result, the effective number of available non-meteor images is:

```text
3,849 images
```

instead of:

```text
3,850 images
```

and:

```text
3,148 images
```

instead of:

```text
3,150 images
```

This does not affect the final target selection, since the difference between 3,850 and 3,849 is minimal, and since
only **1,650 non-meteor images** are required for the balanced dataset.

## Final Dataset Summary

| Item | Value | Target |
|---|---|---:|
| Original dataset size | 7,000 | 7,000 |
| Final dataset size | 5,499 | 5,500 |
| Meteor images kept | 3,849 | 3,850 |
| Meteor images discarded | 1 | 0 |
| Non-meteor images kept | 1,650 | 1,650 |
| Non-meteor images discarded | 1,500 | 1,500 |
| Training set | 4,399 | 4,400 |
| Test set | 550 | 550 |
| Validation set | 550 | 550 |

## Intended Use

This dataset is intended for training, validating, and testing image classification or object detection models designed to distinguish images containing meteors from images without meteors.
