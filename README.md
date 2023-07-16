* Resizing all images to 512x512
* Due to framework used, detections are first converted to normalized grid [-1,1]
* Then for descriptors, converted back to feature coordinates, and gathered.
* Matcher uses temp 20 (different from IMC2022). If I use IMC2022 settings the results are extremely poor.
