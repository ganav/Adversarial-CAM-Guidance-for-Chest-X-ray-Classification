Deep classifiers for chest X-ray diagnosis can be sensitive to the visual frame around the evidence,
producing correct labels for the wrong reasons by relying on background, borders, text marks, 
or acquisition artifacts. This framing sensitivity reduces trustworthiness and can fail under distribution shift.
Inspired by biological vision, especially figure–ground segregation and selective attention, 
we propose a Bio inspired adversarial attention alignment training process that encourages evidence centered decisions without changing the classifier structure.
A classifier is first trained with image level labels. Class activation mapping (CAM) is then used to produce a differentiable heatmap that indicates where the model attends.
We treat this heatmap as a generated localization map and train a discriminator to distinguish generated heatmaps from ground truth masks. 
The classifier is updated using a joint objective that preserves classification performance while pushing its CAM heatmap toward mask like structure, 
reducing reliance on background cues. We also introduce evaluation measures for the test phase, including augmentation inconsistency 
(prediction flip rate under angle based augmentations) and framing sensitivity (CAM energy outside the mask). 
Experiments show improved lung focused attention and robustness,
while requiring masks only during training and no additional inputs at inference.
