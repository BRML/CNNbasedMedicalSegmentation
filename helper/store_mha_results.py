import h5py
import numpy as np
import SimpleITK as sitk
import os
import json

save_dir = os.path.join('..', 'results', 'as_mha', 'brats2013_leaderboard')
file_path = os.path.join('..', 'results', 'as_hdf', 'brats2013_leaderboard_results.hdf5')
seg_file = h5py.File(file_path, 'r')
cropped_seg_maps = seg_file['test_result']

test_set_info_path = os.path.join('..', 'data', 'info', 'brats2013_leaderboard_info.json')
with open(test_set_info_path, 'r') as f:
    info = json.load(f)

if 'sizes' not in info:
    print 'Using default size (240, 240, 155).'
    size_list = [(155, 240, 240)] * len(info['names'])
    info['sizes'] = size_list

for crop_seg, slice_pairs, seg_name, array_size in zip(cropped_seg_maps, info['slices_list'], info['names'], info['sizes']):
    size = (array_size[1], array_size[2], array_size[0])
    full_seg = np.zeros(size, dtype='int8')
    slices = [slice(s[0], s[1]) for s in slice_pairs]
    x_s, y_s, z_s = slices
    full_seg[x_s, y_s, z_s] = crop_seg
    full_seg = np.transpose(full_seg, (2, 0, 1))

    seg_map_itk = sitk.GetImageFromArray(full_seg)
    assert(seg_map_itk.GetSize() == tuple(array_size[::-1]))

    save_path = os.path.join(save_dir, seg_name)
    print 'Saving image as: ', save_path
    sitk.WriteImage(seg_map_itk, str(save_path))
