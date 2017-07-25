from read_images import gen_images, center
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import swirl, rescale, rotate, downscale_local_mean, PiecewiseAffineTransform, warp
from skimage import color
from skimage import filters
import scipy.ndimage as ndi
import scipy.signal as sig
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate as rotate_scipy
from breze.learn.data import one_hot

def groundtruth_(gt):
    """Takes a discrete label volume with zero-indexed labels and applies one_hot encoding."""
    n_classes = gt.max() + 1
    shape = gt.shape
    l = np.reshape(gt, (-1,))
    l = np.reshape(one_hot(l, n_classes), (-1, n_classes))
    gt_onehot = np.reshape(l, shape + (n_classes,))
    return gt_onehot

def rotate_transform(im, gt):
    ang = np.random.uniform(-90, 90)
    axes = np.random.permutation(3)[:2]
    rot_im = rotate_scipy(im, ang, axes=axes, order=3, reshape=False)
    rot_gt = groundtruth_(gt)
    rot_gt = np.array([
        rotate_scipy(class_map, ang, axes=axes, order=3, reshape=False) 
        for class_map in np.transpose(rot_gt, (3, 0, 1, 2))])
    rot_gt = rot_gt.argmax(axis=0)
    rot_gt = np.array(rot_gt, dtype='int8')
    
    return (rot_im, rot_gt)
    
def sinus(image, strength):
    rows, cols = image.shape[0], image.shape[1]

    src_cols = np.linspace(0, cols, 5)
    src_rows = np.linspace(0, rows, 2)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1] - np.sin(np.linspace(0, 2*np.pi, src.shape[0])) * strength 
    dst_cols = src[:, 0]
    dst_rows *= 1.
    dst_rows -= 1.5 * strength
    dst = np.vstack([dst_cols, dst_rows]).T


    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = image.shape[0] #- 1.5 * 5
    out_cols = cols
    out = warp(image, tform, output_shape=(out_rows, out_cols))
    return np.array(out, dtype='float32')

def sinus_(im, strength):
    return np.array([sinus(im_slice, strength) for im_slice in im], dtype='float32')
    
def sinus_transform(im, gt):
    strength = np.random.uniform(3, 6)
    sinus_im = sinus_(im, strength)
    sinus_gt = groundtruth_(gt)
    sinus_gt = np.array([sinus_(class_map, strength) for class_map in np.transpose(sinus_gt, (3, 0, 1, 2))])
    sinus_gt = sinus_gt.argmax(axis=0)
    sinus_gt = np.array(sinus_gt, dtype='int8')
    
    return (sinus_im, sinus_gt)
    
def swirl_(im, strength, radius):
    return np.array([swirl(im_slice, rotation=0, strength=strength, radius=radius) for im_slice in im], dtype='float32')
    
def swirl_transform(im, gt):
    strength = np.random.uniform(1, 2)
    radius = np.random.randint(90, 140)
    
    swirled_im = swirl_(im, strength, radius)
    swirled_gt = groundtruth_(gt)
    swirled_gt = np.array([swirl_(class_map, strength, radius) for class_map in np.transpose(swirled_gt, (3, 0, 1, 2))])
    swirled_gt = swirled_gt.argmax(axis=0)
    swirled_gt = np.array(swirled_gt, dtype='int8')
    
    return (swirled_im, swirled_gt)
    
def rotate_3d_ski(im, gt):
	im = np.transpose(im, (1, 2, 0))
	gt = np.transpose(gt, (1, 2, 0))
	
	ang = np.random.uniform(0, 360)
	r_im = rotate(im , ang, order=3)
	r_gt = rotate(gt, ang, order=3)
	
	return np.transpose(r_im, (2, 0, 1)), np.transpose(r_gt, (2, 0, 1))

def re_rescale(im):
	d_im = zoom(im, (1, 0.5, 0.8), order=3)
	d_im = zoom(d_im, (1, 2, (1/0.8)), order=3)
	
	return d_im

def bounding_box(p1, p2):
	x1, y1, z1 = p1
	x2, y2, z2 = p2
	
	x_s = slice(np.minimum(x1, x2), np.maximum(x1, x2))
	y_s = slice(np.minimum(y1, y2), np.maximum(y1, y2))
	z_s = slice(np.minimum(z1, z2), np.maximum(z1, z2))
	
	return (x_s, y_s, z_s)

def compute_random_region(shape):
	x = np.random.randint(low=0, high=shape[0], size=(2,))
	y = np.random.randint(low=0, high=shape[1], size=(2,))
	z = np.random.randint(low=0, high=shape[2], size=(2,))
	p1 = [c[0] for c in [x, y, z]]
	p2 = [c[1] for c in [x, y, z]]
	
	bb = bounding_box(p1, p2)
	
	return bb

def noise(im, intensity=1, n=1):
	if n > 1:
		new_im = im.copy()
		for i in np.arange(0, n):
			bb = compute_random_region(im.shape)
			new_im[bb] = noise(new_im[bb], intensity, 1)
		return new_im
			
	try:
		noise_vol = np.random.randint(0, int(im.mean()*intensity), size=im.shape)
		noise_vol = np.asarray(noise_vol, dtype='int16')
	except ValueError:
		return im
	return im + noise_vol

def flip(im, full=True):
	if full:
		return im[::-1, ::-1, ::-1]
	else:
		return im[:, ::-1, ::-1]

def sharpen(blurred_im, alpha):
	filter_blurred_im = ndi.gaussian_filter(blurred_im, 1)
	sharp = blurred_im + alpha * (blurred_im - filter_blurred_im)
	
	return sharp
	
def compute_random_shadow(shape, intensity):
	bb = compute_random_region(shape)
	shadow = np.ones(shape, dtype='int16')
	shadow[bb] = intensity
	
	return shadow
	
def shadow(im, intensity=None, n=1):
	if intensity is None:
		intensity = im.mean()
		if im.dtype != 'float':
			intensity = int(intensity)
			
	sh_im = im.copy()
	for i in np.arange(0, n):
		shade = compute_random_shadow(im.shape, intensity)
		sh_im -= shade
	sh_im = np.maximum(0, sh_im)
	
	return sh_im
	
def rotate_3d_scipy(image, gt):
	#if image.dtype != 'float32':
	#	image = np.asarray(image, dtype='float32')
	#if gt.dtype != 'float32':
	#	gt = np.asarray(gt, dtype='float32')

	ang = np.random.uniform(0, 360)
	axes = (1,2)#np.random.permutation(3)[:2]
	rot_im = rotate_scipy(image, ang, axes=axes, order=1, reshape=False)
	rot_gt = rotate_scipy(gt, ang, axes=axes, order=0, reshape=False)
	
	return rot_im, rot_gt

def prep2(gt):
	indices0 = np.where(gt < 0.5)
	indices1 = np.where((gt >= 0.5) & (gt < 1.5))
	indices2 = np.where((gt >= 1.5) & (gt < 2.5))
	indices3 = np.where((gt >= 2.5) & (gt < 3.5))
	indices4 = np.where(gt >= 3.5)
	
	res = np.zeros(gt.shape, dtype='int8')
	res[indices0] = 0
	res[indices1] = 1
	res[indices2] = 2
	res[indices3] = 3
	res[indices4] = 4
	
	return res

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
	
def vis_hems(left, gt_left, right, gt_right):
	left = col_im(left, gt_left)
	right = col_im(right, gt_right)
	
	fig = plt.figure()
	a = fig.add_subplot(1,2,1)
	plt.imshow(left)
	a.set_title('Left Hemisphere')
	a = fig.add_subplot(1,2,2)
	plt.imshow(right)
	a.set_title('Right Hemisphere')
	
	plt.show()
	plt.close()

def vis_diff_modalities(*ims):
	flair, t1, t1_c, t2, gt = ims
	flair, t1, t1_c, t2 = [col_im(x, gt) for x in ims[:-1]]
	fig = plt.figure()
	a = fig.add_subplot(2,2,1)
	plt.imshow(flair)
	a.set_title('Flair')
	plt.axis('off')
	a = fig.add_subplot(2,2,2)
	plt.imshow(t1)
	a.set_title('T1')
	plt.axis('off')
	a = fig.add_subplot(2,2,3)
	plt.imshow(t1_c)
	a.set_title('T1c')
	plt.axis('off')
	a = fig.add_subplot(2,2,4)
	plt.imshow(t2)
	a.set_title('T2')
	plt.axis('off')
	
	plt.show()

def show_brains():
	for im in gen_images(n=-1):
		t_im = im['T1c']
		gt = im['gt']
		
		for _slice in np.arange(0, t_im.shape[0], t_im.shape[0]/15):
			im_slice = t_im[_slice]
			gt_slice = gt[_slice]
			
			vis_col_im(im=im_slice, gt=gt_slice)
			
def show_modalities():
	for im in gen_images(n=-1, crop=True):
		ims = [im['Flair'], im['T1'], im['T1c'], im['T2'], None]
		for _slice in np.arange(0, ims[0].shape[0], ims[0].shape[0]/20):
			im_slices = [x[_slice] if x is not None else x for x in ims]
			
			vis_diff_modalities(*im_slices)

def show_downsize():
	for im in gen_images(n=-1, crop=True):
		t_im = im['T1c']
		gt = im['gt']
		
		t_im = np.asarray(t_im, dtype='float32')
		gt = np.asarray(gt, dtype='float32')
		
		d_im = zoom(t_im, 0.5, order=3)
		d_gt = zoom(gt, 0.5, order=0)
		print 'New shape: ', d_im.shape
		
		slices1 = np.arange(0, d_im.shape[0], d_im.shape[0]/20)
		slices2 = np.arange(0, t_im.shape[0], t_im.shape[0]/20)
		
		for s1, s2 in zip(slices1, slices2):
			d_im_slice = d_im[s1]
			d_gt_slice = d_gt[s1]
			
			im_slice = t_im[s2]
			gt_slice = gt[s2]
			
			title0= 'Original'
			title1= 'Downsized'
			vis_ims(im0=im_slice, gt0=gt_slice, im1=d_im_slice, 
				gt1=d_gt_slice, title0=title0, title1=title1)

def show_crops():
	x_c = 119
	y_c = 119
	z_c = 77
	
	count = 1
	for im in gen_images(n=-1, crop=True):
		print 'image %i: ' % count
		t_im = im['T1c']
		gt = im['gt']
		print t_im.shape
		for _slice in np.arange(0, t_im.shape[0], t_im.shape[0]/20):
			im_slice = t_im[_slice]
			gt_slice = gt[_slice]
			
			vis_col_im(im=im_slice, gt=gt_slice)
		count += 1
			
def show_hemisphere():
	x_c = 119
	y_c = 119
	z_c = 77
	
	for im in gen_images(n=-1, crop=True):
		t_im = im['T1c']
		gt = im['gt']
		
		left = t_im[:,:,:t_im.shape[-1]/2]
		gt_left = gt[:,:,:gt.shape[-1]/2]
		
		right = t_im[:,:,t_im.shape[-1]/2:]
		gt_right = gt[:,:,gt.shape[-1]/2:]
		
		for _slice in np.arange(0, t_im.shape[0], t_im.shape[0]/20):
			l_slice = left[_slice]
			gt_l_slice = gt_left[_slice]
			r_slice = right[_slice]
			gt_r_slice = gt_right[_slice]
			
			vis_hems(left=l_slice, gt_left=gt_l_slice, right=r_slice, gt_right=gt_r_slice)
			
def show_rotation():
	for im in gen_images(n=-1, crop=True):
		t_im = im['T1c']
		gt = im['gt']
		
		rot_im, rot_gt = rotate_3d_scipy(t_im, gt)
		rot_gt = np.asarray(rot_gt, dtype='int8')
		#rot_gt = prep2(rot_gt)
		for _slice in np.arange(0, rot_im.shape[0], rot_im.shape[0]/20):
			im_slice = rot_im[_slice]
			gt_slice = rot_gt[_slice]
			
			vis_col_im(im=im_slice, gt=gt_slice)
			
def show_transform():
    for im in gen_images(n=-1, crop=True):
        t_im = im['T1c']
        gt = im['gt']
        #t_im_trans, trans_gt = rotate_transform(t_im, gt)
        #t_im_trans = t_im
        #t_im_trans = re_rescale(t_im)
        #t_im_trans = flip(t_im)
        #t_im_trans = noise(t_im, intensity=1, n=10)
        t_im_trans, trans_gt = ndi.percentile_filter(t_im, np.random.randint(0, 10), (2, 2, 2)), gt
        #t_im_trans = ndi.morphological_gradient(t_im, size=(2, 2, 2))
        #t_im_trans = ndi.grey_dilation(t_im, size=(3, 3, 3))
        #t_im_trans = ndi.grey_erosion(t_im_trans, size=(3, 3, 3))
        
        print t_im_trans.dtype
        
        for _slice in np.arange(0, t_im.shape[0], t_im.shape[0]/20):
            im_slice = t_im[_slice]
            im_slice_trans = t_im_trans[_slice]
            gt_slice = gt[_slice]
            trans_gt_slice = trans_gt[_slice]
            
            vis_ims(im0=im_slice, gt0=gt_slice, im1=im_slice_trans, gt1=trans_gt_slice)
        
if __name__ == '__main__':
	#show_brains()
	#show_modalities()
	#show_downsize()
	show_crops()
	#show_hemisphere()
	#show_rotation()
	#show_transform()