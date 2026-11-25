# TrashNet-custom v2 (6 waste classes)

Base: TrashNet (Thung & Yang, 2016; github.com/garythung/trashnet, `dataset-resized`; repository MIT licence, citation requested).
Extra images: Garbage Dataset (GD), S. Kunwar, "The Garbage Dataset (GD): A Multi-Class Image Benchmark for Automated Waste Segregation", arXiv:2602.10500; Kaggle sumn2u/garbage-classification-v2, licence shown on Kaggle: MIT. GD says its images come from phone photos, public repositories, web scraping and community submissions, so the rights to individual images are not guaranteed by that label.
All images are 256x192 JPEG (GD images were centre-cropped to 4:3 first).

## Splits (counts in build_stats.json, per-file info in manifest.csv; column `dataset` = trashnet / garbage_dataset_kaggle)
- `train/`  TrashNet originals + synthetic background-replaced images (`*_bg*_<kind>.jpg`) + up to 400 new GD images per class.
- `val/`    TrashNet originals + 40 new GD images per class.
- `test/`   TrashNet originals only (in-distribution test).
- `test_external/`  80 GD images per class, from a different source, never in train/val (external test set).
- `val_bgshift/`, `test_bgshift/`  TrashNet val/test photos with synthetic background replacement (robustness checks, never trained on).

## Cleaning and construction
1. TrashNet: removed 6 images appearing under two labels; removed 21 near-duplicates (same class, pHash distance <= 6 and pixel correlation >= 0.95); stratified 70/15/15 split (seed 42).
2. Synthetic backgrounds (train only): object cut-out (GrabCut) pasted on procedural backgrounds with random flip/rotation/scale, shadow and photometric jitter; only cut-outs passing automatic quality checks are used; "trash" gets 4 variants per accepted photo, other classes 1.
3. GD: 2,218 of the 7,463 images in the six matching classes are copies of TrashNet images (pHash <= 6 and correlation >= 0.95) and were excluded, together with 1 further close match and 12 within-GD near-duplicates. Only the remaining novel images are used.
4. Leakage checks (pHash + pixel correlation): 0 near-duplicates between test_external and train/val/test, 0 between test and train, 0 between val and train.

## Limitations
- Synthetic backgrounds are procedural textures, and cut-outs are approximate (1,388 of 2,500 TrashNet photos were rejected by the quality check).
- GD is not independent of TrashNet (30% overlap found), and may overlap with other public datasets we did not check. The external test set is a different collection, not guaranteed real-world data.
- GD labels were not audited; GD images have busy backgrounds and uneven lighting according to its authors.
- Synthetic "trash" images come from only 72 source photos.
