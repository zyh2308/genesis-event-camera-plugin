# E-POSE sample access check

Checked from WSL at `2026-10-16T06:53:22Z`; no model inference was run.

## Evidence

- DataCite DOI metadata request succeeded (`HTTP 200`) for `10.6084/m9.figshare.28106717`. It identifies *E-POSE: A Large Scale Event Camera Dataset for Object Pose Estimation*, publication year 2025, CC-BY-4.0, and a total registered size of 29,759,153,475 bytes. The landing URL is `https://springernature.figshare.com/articles/dataset/E-POSE_A_Large_Scale_Event_Camera_Dataset_for_Object_Pose_Estimation/28106717`.
- A GET of that official Springer Nature Figshare landing page returned `HTTP 202`, `x-amzn-waf-action: challenge`, and an empty body.
- A GET of the public Figshare article API endpoint `https://api.figshare.com/v2/articles/28106717` returned `HTTP 403` with a short HTML body.
- The DataCite record contains no file-level manifest, so it does not disclose a file ID that could bypass the blocked landing/API response. No raw sample, corresponding CAD, category record, K matrix, or pose annotation was obtained in this run.

Raw request metadata and bodies are retained in this directory. This is a current network/WAF access blocker, not evidence that the published dataset is unavailable. The model/CAD contract remains unverified; no generic cuboid is represented as an E-POSE match. Recommended next action for the research session: obtain an authorized publisher download route, author-provided mirror, or a sample plus its matched CAD, then perform the registered raw event/pose/K/coordinate/scale audit before any model integration decision.
