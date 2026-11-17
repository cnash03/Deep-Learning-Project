# Q&A prep for the Peer Showcase

Short answers to likely classmate questions. Each answer names the teammate best placed to give it. Every number below comes from `results/eda/eda_summary.md` or the paper's Methods section. Items marked **[TBD]** depend on the training results. Don't post an answer that still contains a TBD.

---

**1. Why CNNs and not a transformer like SegFormer?** *(Callen)*
Our question was a controlled one: decoder, encoder depth, loss, and pretraining, changing one at a time. We wanted that comparison on well-understood baselines that fit our compute budget (about 14 runs). The RUGD paper's own baselines are ResNet-50 CNNs, so our numbers have a reference point. A transformer backbone (SegFormer, or a GA-Nav-style model) is the first item on our future-work list. It might do better on wide context such as sky/tree boundaries, but we didn't test it, so we can't claim it would.

**2. Why 384×384 instead of 512×512, as in the proposal?** *(Callen)*
Compute. In our benchmark, a 512² step was about 1.9× slower than a 384² step, and the full matrix is about 14 runs. We kept 384² and 25 epochs for *every* run, so all comparisons stay fair, and we disclose this as a deviation from the proposal. The cost is lost detail on thin structures such as poles, signs and log edges. Higher resolution, or sliding-window inference, is the first next step.

**3. Why leave *bridge* out of mIoU? Isn't that cherry-picking?** *(Xander)*
*Bridge* has 0 training pixels and about 4.6 million test pixels, all from the *creek* sequence. No model trained on the official split can predict it. Its IoU is 0 by construction for every method, including all the published RUGD baselines. Leaving it in lowers every model's mIoU by the same amount and tells you nothing about the methods. We still report bridge's per-class IoU, and the evaluation code also outputs mIoU with bridge included: [TBD: value]. The rule was set in the config before we saw any results.

**4. Why do you expect validation to overstate test?** *(Xander)*
The splits have different class distributions. Safety classes are 1.1% of training pixels, 1.2% of validation and 10.3% of test. Rock-bed is 114× more common in test than in training, and 98% of it comes from one test sequence (*creek*). Validation also has no sand, rock-bed, bicycle or bridge pixels at all. The total-variation distance between class distributions is 0.11 for train–val but 0.25 for val–test. We measured a validation-to-test gap of [TBD] mIoU points.

**5. Would this work on a full-size truck?** *(Xander)*
Not without validation on truck data. RUGD's camera sits under 25 cm off the ground, so a truck's camera would see the same terrain from a much higher, steeper angle. RELLIS-3D comes from a larger vehicle, which makes our zero-shot test a partial proxy: group mIoU went from [TBD] to [TBD]. Changes in viewpoint, geography and season would all need their own data. And as a research model, it isn't meant to drive anything.

**6. How would you improve the rare classes further?** *(Callen)*
Options we'd try, roughly in this order:
- Higher input resolution, since water edges and logs are thin.
- Class-aware sampling that oversamples frames containing water or rock (water appears in only about 1 in 13 training frames).
- Losses aimed at IoU directly, such as Lovász-softmax.
- Temporal consistency across neighboring video frames.
- More labeled hazard data. Water and rock-bed are concentrated in only a few sequences, which no loss function can fully fix.

[TBD: mention which of our losses helped most.]

**7. Why not fine-tune on RELLIS-3D?** *(Xander)*
The stretch goal was to measure how far a RUGD model transfers *as is*, and fine-tuning would answer a different question. Fine-tuning on a small labeled RELLIS-3D subset is on our future-work list, because it shows how much target data closes the gap. Note also that RELLIS-3D is licensed CC BY-NC-SA, so anything trained on it is for non-commercial use only.

**8. What biases does the dataset carry?** *(Xander or Callen)*
- One geographic area (parks, trails, a creek, and a village edge), recorded by one small robot.
- Narrow coverage of weather, season and time of day. There's no snow, desert or night, and no *mud* class.
- Labels drawn as coarse polygons, with judgment calls between dirt, sand, gravel, mulch and concrete.
- People are only 0.012% of training pixels, so person IoU must not be read as evidence of pedestrian safety.

We report per-class and per-sequence results so these gaps stay visible.

**9. How do you know you didn't overfit to the test set?** *(Callen)*
A script picked every architecture, loss and seed run from validation mIoU alone, and logged each decision to `results/decisions.json`. The evaluation code refuses to touch the test split without an explicit `--confirm-test` flag. We evaluated test once per final configuration, after selection was finished. We split by video sequence, so near-duplicate consecutive frames never appear in both training and test.

**10. Are your differences bigger than run-to-run noise?** *(Callen)*
We reran the top two configurations with three seeds each. The seed standard deviation is [TBD] mIoU points, and [TBD: which comparisons exceed it]. Three seeds give only a rough bound, which we state as a limitation.

---

**Forum tips**
- Answer within a day, in 3–5 sentences, and link to the paper section or repo file that backs the answer.
- If a question exposes a real weakness, say so and say what you would do about it.
- Requirement: respond to at least 2 classmate questions or comments. Both of us should post answers.
