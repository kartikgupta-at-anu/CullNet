import yaml
import numpy as np

with open("../models/testing/LINEMOD_singleobj_realsyn.yml", 'r') as stream:
    res_data = yaml.load(stream)

np.mean(res_data['99']['ADD'].values())
np.mean(res_data['99']['Reprojection_metric'].values())

with open("../models/testing/LINEMOD_singleobj_realsyn_topk6_nearby1_non_nms.yml", 'r') as stream:
    res_data = yaml.load(stream)

np.mean(res_data['115']['ADD'].values())
np.mean(res_data['115']['Reprojection_metric'].values())

with open("../models/testing/LINEMOD_singleobj_realsyn_topk6_nearby1_cull_modebias.yml", 'r') as stream:
    res_data = yaml.load(stream)

np.mean(res_data['115']['ADD'].values())
np.mean(res_data['115']['Reprojection_metric'].values())
