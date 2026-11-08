# Registered depth-only normal-teacher projection short study

Registration is made after the frozen synthetic factor/gradient diagnostic and before any candidate optimizer update. This is one downstream optimization ablation shared by EventGenesis and V2E, not a simulator change.

## Evidence and decision

The matching four-rollout,64-window normal/half/eighth export passed54 original native-summary checks, with zero updates and unchanged source/weights. Removing an apparent 96-rollout-versus4-rollout illumination confound did not remove the half/dark tradeoff. Existing ordered factor results show the EG C1 half regression already loses6.77pp when only anchor depth changes; center and crop-dependent rotation add further loss. The cross-generator dark center gap (~29px vs16px) remains a separate problem.

Prespecified raw count-gain0.5/2 changes EG C1 center by only~0.5-0.7px and depth~0.22-0.40cm. Prediction-only local3 decode worsens EG C1 dark localization. The same-geometry global-context swap affects EG depth less than V2E, so neither an untested normalization nor a decoder change is adopted.

The fixed32 TRAIN windows, three seeds and complete24-parameter depth space show physical-supervision vs weighted2*normal-teacher cosine [-0.951,-0.975,-0.946] for EG C1. Their summed-gradient cancellation ratios are [.161,.142,.238]; two half-task directions are negative. V2E teacher gradients are much smaller relative to physical supervision. Center gradients retain positive physical alignment. These are local gradients at frozen endpoints, not observed historical AdamW steps; reverse regularization gradients can represent equilibrium. They justify a bounded intervention test, not a root-cause proof or improvement claim.

Selected treatment: preserve the physical gradient P; remove only the component of the already weighted teacher gradient T that opposes P, in one full-model space: Tprime=T-min(dot(T,P)/dot(P,P),0)*P; if P=0 retainT. Use G=P+Tprime, then original norm5 clipping and AdamW. No per-layer coefficient, epsilon, threshold search or light-specific rule. Normal teacher weight remains2, auxiliary half/eighth ratio3:1, features/heads LR1.5e-4, original physical supervised loss and regularization decay unchanged. The one-sided modification is inspired by PCGrad's projection mechanism, not a claim to reproduce its symmetric multi-task algorithm. [Paper](https://arxiv.org/abs/2001.06782), [author implementation](https://github.com/tianheyu927/PCGrad/blob/master/PCGrad_tf.py).

Alternatives not selected: removing all teacher supervision would also remove nonconflicting retention; changing LR or input amplitude was not supported as the main mechanism. The independent center branch stays at each method/seed's already verified C1 checkpoint. This intervention deliberately tests the depth mechanism; it does not assert that depth alone will close the separate dark localization gap.

## Fixed experiment

- Two methods, seeds0/1/2; same native physical bank, labels, model lineage and exact C1 sample/augmentation replay.
- Start each depth branch from the original common source state used for C1, not from the later slow-feature checkpoint.
- Six new depth fits, each288 updates/sixepochs/batch32:1728 new updates. Reuse24 C0/C1 control fits and6 C1 candidate-center branches without updates. Record36 branch records with source method/seed/arm explicitly; do not count reused records as new training.
- Preserve the original native main/evaluation, full clean confirmation,30 illumination checks and8 confirmations. Report all three light conditions and seeds, both generators and C0/C1/candidate. Do not adopt or open new E-POSE scores if the original stability checks fail; compare the predefined EG/V2E gaps without selecting favorable samples.
- Fixed candidate maps into the native paired_consistency legacy slot only for unchanged evaluation. Actual auxiliary teacher weight is0; candidate canonical name teacher_projection.
- Smoke: seed0, both methods; each C0/C1 depth/center1 update and candidate depth1; candidate center reuses smoke C1 center. Total10 actual updates, plus independent model/Adam restoration, original source teacher hash, exact paired label/augmentation hashes, finite gradients and projection formula check. Smoke is not quality evidence.
- Raw loss logging must distinguish the original scalar objective from the projected optimizer direction. Save projection statistics and never claim the applied direction equals ordinary backprop of that scalar.

## Safety and isolation

Frozen core62da4f35ce3c5c96f04268bd1687a27657af9238, physical timestamps, noise and occlusion unchanged. Newreal scoring closed; all labels and gradient probes are synthetic physical truth. Use only assigned g1 GPU4 after diagnostics and smoke. Immutable snapshots, exact UUID/source pins and unique roots; retain failures. Do not interrupt reconstruction, Census or fixed MVSEC jobs, and do not modify their sources/results.
