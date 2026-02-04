Road Audit Tile Labels (tile=128px)

Files:
- labels.csv : one row per tile
- tiles/     : 128x128 jpg tiles referenced by labels.csv

Label mapping (from labels.csv):
- 0 = clean_road   (no obvious litter/debris; shadows/markings/stains still count as clean)
- 1 = unclean_road (visible litter/debris/dirt clumps/gravel/leaf/plastic/paper etc)
- 2 = ignore       (tile not mainly road OR can't judge due to blur/overexposure/occlusion)

Recommended split:
- Split by video_id (NOT random tiles) to avoid near-duplicate leakage.

