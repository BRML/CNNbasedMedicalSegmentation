import os
import six
import cPickle as pickle
import SimpleITK as sitk
import numpy as np
import warnings
from breze.learn.data import one_hot
import h5py
from scipy.ndimage import zoom
from find_mha_files import get_patient_dirs
from random import shuffle
from skimage import color
import copy
import matplotlib.pyplot as plt

def vis_col_im(im, gt):
	indices_0 = np.where(gt == 0) # nothing
	indices_1 = np.where(gt == 1) # necrosis
	indices_2 = np.where(gt == 2) # edema
	indices_3 = np.where(gt == 3) # non-enhancing tumor
	indices_4 = np.where(gt == 4) # enhancing tumor
	
	im = np.asarray(im, dtype='float32')
	im = im*1./im.max()
	rgb_image = color.gray2rgb(im)
	m0 = [1., 1., 1.]
	m1 = [1., 0., 0.]
	m2 = [0.2, 1., 0.2]
	m3 = [1., 1., 0.2]
	m4 = [1., 0.6, 0.2]
	
	im = rgb_image.copy()
	im[indices_0[0], indices_0[1], :] *= m0
	im[indices_1[0], indices_1[1], :] *= m1
	im[indices_2[0], indices_2[1], :] *= m2
	im[indices_3[0], indices_3[1], :] *= m3
	im[indices_4[0], indices_4[1], :] *= m4
	
	plt.imshow(im)
	plt.show()
	plt.close()

def col_im(im, gt):
	im = np.asarray(im, dtype='float32')
	im = im*1./im.max()
	rgb_image = color.gray2rgb(im)
	im = rgb_image.copy()
	
	if gt is None:
		return im
		
	indices_0 = np.where(gt == 0) # nothing
	indices_1 = np.where(gt == 1) # necrosis
	indices_2 = np.where(gt == 2) # edema
	indices_3 = np.where(gt == 3) # non-enhancing tumor
	indices_4 = np.where(gt == 4) # enhancing tumor
	
	m0 = [1., 1., 1.]
	m1 = [1., 0., 0.] # red: necrosis
	m2 = [0.2, 1., 0.2] # green: edema
	m3 = [1., 1., 0.2] # yellow: non-enhancing tumor
	m4 = [1., 0.6, 0.2] # orange: enhancing tumor
	
	im[indices_0[0], indices_0[1], :] *= m0
	im[indices_1[0], indices_1[1], :] *= m1
	im[indices_2[0], indices_2[1], :] *= m2
	im[indices_3[0], indices_3[1], :] *= m3
	im[indices_4[0], indices_4[1], :] *= m4
	
	return im

def vis_ims(im0, gt0, im1, gt1, title0='Original', title1='Transformed'):
	im0 = col_im(im0, gt0)
	im1 = col_im(im1, gt1)
	
	fig = plt.figure()
	a = fig.add_subplot(1,2,1)
	plt.imshow(im0)
	a.set_title(title0)
	a = fig.add_subplot(1,2,2)
	plt.imshow(im1)
	a.set_title(title1)
	
	plt.show()
	plt.close()
    
def get_im_as_ndarray(image, downsize=False):
    ims = [image['Flair'], image['T1'], image['T1c'], image['T2']]
    if downsize:
        ims = [zoom(x, 0.5, order=1) for x in ims]
    im = np.array(ims, dtype='int16')
	
    return im

def get_gt(gt, n_classes, downsize=False):
    if not downsize:
        return gt
    original_shape = gt.shape
    gt_onehot = np.reshape(gt, (-1,))
    gt_onehot = np.reshape(one_hot(gt_onehot, n_classes), original_shape + (n_classes,))
    gt_onehot = np.transpose(gt_onehot, (3, 0, 1, 2))
    
    zoom_gt = np.array([zoom(class_map, 0.5, order=1) for class_map in gt_onehot])
    zoom_gt = zoom_gt.argmax(axis=0)
    zoom_gt = np.asarray(zoom_gt, dtype='int8')
    
    return zoom_gt

def convert_gt_to_onehot(gt, n_classes):
    gt_onehot = np.transpose(gt, (1, 2, 0))
    gt_onehot = np.reshape(gt_onehot, (-1,))
    gt_onehot = np.reshape(one_hot(gt_onehot, n_classes), (-1, n_classes))
    
    return gt_onehot
    
def process_gt(gt, n_classes, downsize=False):
	if downsize:
		gt = zoom(gt, 0.5, order=0)
		gt = np.asarray(gt, dtype='int8')
	gt = np.transpose(gt, (1, 2, 0))
	l = np.reshape(gt, (-1,))
	l = np.reshape(one_hot(l, n_classes), (-1, n_classes))
	return l
    
def center(im):
	indices = np.where(im > 0)
	indices = np.array(indices)
	indices = indices.T
	
	return [int(i) for i in np.round(np.mean(indices, axis=0))]

def get_pats(dir_name):
	pats = []
	pats = get_patient_dirs(dir_name, pats)
	return pats

def find_mha_paths(pat_dir, paths):
	for item in os.listdir(pat_dir):
		item_path = os.path.join(pat_dir, item)
		if os.path.isdir(item_path):
			paths = find_mha_paths(item_path, paths)
		elif os.path.isfile(item_path):
			if item.endswith('.mha'):
				paths.append(item_path)
	return paths

def get_im(pat_dir):
	paths = []
	paths = find_mha_paths(pat_dir, paths)
	gt = None
	im = {'Flair': None, 'T1': None, 'T1c': None, 'T2': None, 'gt': None}
	for p in paths:
		itk_image = sitk.ReadImage(p)
		nd_image = sitk.GetArrayFromImage(itk_image)
		if 'more' in p or 'OT' in p:
			if gt is None:
				gt = nd_image
			else:
				raise ValueError('Found multiple ground truths.')
		elif 'Flair' in p:
			im['Flair'] = nd_image
		elif 'T1c' in p:
			im['T1c'] = nd_image
		elif 'T1' in p:
			im['T1'] = nd_image
		elif 'T2' in p:
			im['T2'] = nd_image
		else:
			print 'Unexpected path: ', p
	if gt is None:
		warnings.warn('Could not find ground truth. Is this a test image?')
	im['gt'] = gt
	return im	

def check(coords, shape):
	z, y, x = coords
	sl_z = (z-64, z+64)
	sl_y = (y-84, y+76) # -70, +90
	sl_x = (x-72, x+72)
	if sl_z[0] < 0:
		sl_z = (0, 128)
	elif sl_z[1] > shape[0]:
		sl_z = (shape[0]-128, shape[0])
		
	if sl_y[0] < 0:
		sl_y = (0, 160)
	elif sl_y[1] > shape[1]:
		sl_y = (shape[1]-160, shape[1])
		
	if sl_x[0] < 0:
		sl_x = (0, 144)
	elif sl_x[1] > shape[2]:
		sl_x = (shape[2]-144, shape[2])
		
	z_s = slice(sl_z[0], sl_z[1])
	y_s = slice(sl_y[0], sl_y[1])
	x_s = slice(sl_x[0], sl_x[1])
	
	return (z_s, y_s, x_s)
	
def create_folds(dir_name='..//BRATS2015_Training'):
    pats = get_pats(dir_name)
    shuffle(pats)
    
    folds = []
    for i in range(3):
        validation_slice = slice(i*74, i*74+74)
        valid_and_test = pats[validation_slice]
        valid = valid_and_test[:37]
        test = valid_and_test[37:]
        train = pats[:i*74] + pats[i*74+74:]
        folds.append({
            'train': copy.deepcopy(train),
            'valid': copy.deepcopy(valid),
            'test': copy.deepcopy(test)
        })
        
    with open('folds.pkl', 'w') as f:
        pickle.dump(folds, f)
        
    return folds
    
def test_folds():
    folds = create_folds()
    
    for i, fold in enumerate(folds):
        print 'Fold %i: ' % (i+1)
        for key in ['train', 'valid', 'test']:
            print '\t%s: ' % key
            print '\t%i patients' % len(fold[key])
            for patient in fold[key]:
                print '\t', patient
                
def build_hdf5_from_fold():
	"""
	Function for creating our training set.
	This function will first search a file called
	folds.pkl in the current directory. folds.pkl
	is a file detailing three cross-validation folds,
	where each cross-validation fold has a different
	set of training, validation and testing partitions.
	If no folds.pkl is present, the function create_folds
	will be called to create it. The variable fold_number
	determines which of the three cross-validation folds should 
	be used to create the data set.
	The data set itself will be a .hdf5 file that will be saved under
	../data/datasets/brats_foldX.hdf5 where X=fold_number
	"""
	if os.path.exists('folds.pkl'):
		with open('folds.pkl', 'r') as f:
			folds = pickle.load(f)
	else:
		folds = create_folds('BRATS2015_Training')
		
	fold_number = 0 # ADAPT TO FOLD
	fold = folds[fold_number]
	
	if not os.path.exists('data//datasets'):
		os.makedirs('data//datasets')
		
	data = h5py.File('data//datasets//brats_fold'+str(fold_number)+'.hdf5', 'w')
	depth, height, width = (128, 160, 144)
	n_chans = 4
	dimprod = height*width*depth
	n_classes = 5
	
	train_size = 200
	valid_size = 37
	test_size = 37
	
	x = data.create_dataset('train_x', (train_size, depth, n_chans, height, width), dtype='int16')
	vx = data.create_dataset('valid_x', (valid_size, depth, n_chans, height, width), dtype='int16')
	tx = data.create_dataset('test_x', (test_size, depth, n_chans, height, width), dtype='int16')
	y = data.create_dataset('train_y', (train_size, dimprod, n_classes), dtype='int8')
	vy = data.create_dataset('valid_y', (valid_size, dimprod, n_classes), dtype='int8')
	ty = data.create_dataset('test_y', (test_size, dimprod, n_classes), dtype='int8')

	dat_access = {
		'train': (x, y),
		'valid': (vx, vy),
		'test': (tx, ty)
	}
	
	for key in ['train', 'valid', 'test']:
		print 'building %s set' % key
		index = 0
		size = len(fold[key])
		for image in gen_images(custom_pats=fold[key], crop=True, n=-1):
			print '\treading image %i of %i...' % (index+1, size)
			gt = get_gt(image['gt'], n_classes, downsize=False)
			im = get_im_as_ndarray(image, downsize=False)
			
			# sanity check
            #t_im = im[0]
            #for _slice in np.arange(0, t_im.shape[0], t_im.shape[0]/15):
            #    im_slice = t_im[_slice]
            #    gt_slice = gt[_slice]   
            #    vis_ims(im0=im_slice, gt0=gt_slice, im1=im_slice, gt1=np.zeros(im_slice.shape))
            #
			
			dat_access[key][0][index, :, :, :, :] = np.transpose(im, (1, 0, 2, 3))
			dat_access[key][1][index, :, :] = convert_gt_to_onehot(gt, n_classes)
			index += 1		
	data.close()

def get_image_slice(image):
	z_s, x_s, y_s = check(center(image['Flair']), image['Flair'].shape)
	im = {}
	for key, value in six.iteritems(image):
		if value is not None:
			im.update({key: value[z_s, x_s, y_s]})
	return im, (z_s, x_s, y_s)


def gen_images(dir_name='..//BRATS2015_Training', n=1, specific=False, interval=None, crop=False, randomize=False, custom_pats=None):
	pats = get_pats(dir_name) if custom_pats is None else custom_pats
	print '%i images in total.' % len(pats)
	if randomize:
		print 'shuffling patients.'
		shuffle(pats)
	
	im_gts = []
	if interval is None:
		a = 0
		b = n
	else:
		a, b = interval
	if b == -1:
		b = len(pats)
	if a == -1:
		pats = pats[::-1]
		a = 0
		print 'yielding images in reverse order.'
	elif b > len(pats):
		raise ValueError('There are %i images but user requested %i.' % (len(pats), b))
	if not specific:
		print 'yielding images in range: (%i, %i).' % (a, b)
		for p in pats[a:b]:
			try:
				print('{}\t'.format(p))
				im = get_im(p)
			except ValueError:
				print 'Problem with: ', p
				raise
			if im is not None:
				if not crop:
					yield im
				else:
					z_s, x_s, y_s = check(center(im['Flair']), im['Flair'].shape)
					for key in im:
						if im[key] is not None:
							im[key] = im[key][z_s, x_s, y_s]
					yield im
	else:
		if b != len(pats):
			print 'yielding image %i.' % b
			p = pats[b]
			try:
				im = get_im(p)
			except ValueError:
				print 'Problem with: ', p
				raise
			if not crop:
				yield im
			else:
				for key in im:
					z_s, x_s, y_s = check(center(im['Flair']), im['Flair'].shape)
					if im[key] is not None:
						im[key] = im[key][z_s, x_s, y_s]
				yield im
		else:
			raise ValueError('There are %i images but user requested image %i(images are zero-indexed).' % (len(pats), b))
		
def make_data_set():
	data = h5py.File('data.hdf5', 'w')
	depth, height, width = (64, 80, 72)
	n_chans = 4
	dimprod = height*width*depth
	n_classes = 5
	
	train_size = 200
	valid_size = 37
	test_size = 37
	
	x = data.create_dataset('train_x', (train_size, depth, n_chans, height, width), dtype='int16')
	vx = data.create_dataset('valid_x', (valid_size, depth, n_chans, height, width), dtype='int16')
	tx = data.create_dataset('test_x', (test_size, depth, n_chans, height, width), dtype='int16')
	y = data.create_dataset('train_y', (train_size, dimprod, n_classes), dtype='int8')
	vy = data.create_dataset('valid_y', (valid_size, dimprod, n_classes), dtype='int8')
	ty = data.create_dataset('test_y', (test_size, dimprod, n_classes), dtype='int8')

	dat_access = {
		'train': (x, y),
		'valid': (vx, vy),
		'test': (tx, ty)
	}
	
	count = 0
	index = 0
	access_code = 'train'
	print 'starting with train set'
	for image in gen_images(n=-1, crop=True, randomize=True):
		if count == 274:
			print 'read 274 images, terminating...'
			break
		print '\tReading image %i...' % (count+1)
		gt = process_gt(image['gt'], n_classes, downsize=True)
		im = get_im_as_ndarray(image, downsize=True)
		
		dat_access[access_code][0][index, :, :, :, :] = np.transpose(im, (1, 0, 2, 3))
		dat_access[access_code][1][index, :, :] = gt
		
		index += 1
		count += 1
		if count == 200:
			print 'train set complete, proceeding to valid set.'
			access_code = 'valid'
			index = 0
		elif count == 237:
			print 'valid set complete, proceeding to test set.'
			access_code = 'test'
			index = 0
	data.close()

def get_shapes(im):
	shapes = []
	for key in im:
		shapes.append(im[key].shape)
	return shapes
	
def check_shapes(im):
	for sh in get_shapes(im):
		if sh != (128, 160, 144):
			return False
	return True
	
def test_shapes():
	count = 1
	errors = 0
	for image in gen_images(n=-1, crop=True):
		if not check_shapes(image):
			print 'Problem with image %i.' % count
			errors += 1
		else:
			print 'image %i is ok.' % count
		count += 1
	print 'Finished with %i errors.' % errors
	
if __name__ == '__main__':
    #make_data_set()
    #test_shapes()
    #test_folds()
    #create_folds()
    build_hdf5_from_fold()
