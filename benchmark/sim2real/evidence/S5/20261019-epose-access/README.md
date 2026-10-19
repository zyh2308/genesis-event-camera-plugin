# E-POSE public-access check

This audit was performed on 2026-10-19. The official Figshare API returned 403; the article page returned 202 with an empty body; and the direct article downloader returned 403. I did not retry these endpoints in a loop. The official E-POSE GitHub API was reachable and lists 25 mesh files across 13 object-name folders, but contains no raw event HDF5 or per-sample MAT annotations. The model-to-sample match therefore remains unverified.

The paper states that raw data are HDF5 and pose annotations are in MAT files; the official code README says its existing scripts currently read bag files and have not yet been adapted to the provided HDF5. We still need one named original HDF5 sequence and matching MAT file to check timestamps, K, coordinate frames, CAD scale, and symmetry. Generic-cuboid synthetic checks remain mechanism-only and cannot stand in for E-POSE evaluation.

Sources: [E-POSE paper](https://doi.org/10.1038/s41597-025-04536-5), [official data record](https://doi.org/10.6084/m9.figshare.28106717), [official code and geometry repository](https://github.com/oussamaabdulhay/E_Pose).
