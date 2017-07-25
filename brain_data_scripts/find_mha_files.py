import os
import sys

def find_patient_dirs(dirs):
	patients = []
	for d in dirs:
		pat_code = d['path'].split('VSD')[0]
		if pat_code not in patients:
			patients.append(pat_code)
	return patients	

def get_patient_dirs(path, dirs):
	for it in os.listdir(path):
		it_path = os.path.join(path, it)
		if os.path.isdir(it_path):
			if 'pat' in it:
				dirs.append(it_path)
			else:
				dirs = get_patient_dirs(it_path, dirs)
	return dirs

def _crawl(path, dirs):
	files = []
	mhas = []
	for it in os.listdir(path):
		it_path = os.path.join(path, it)
		if os.path.isfile(it_path):
			if it.endswith('.mha'):
				mhas.append(it_path)
			else:
				files.append(it_path)
		elif os.path.isdir(it_path):
			dirs = _crawl(it_path, dirs)
	if len(mhas) > 0 or len(files) > 0:
		new_dir = {'path': path, 'files': files, 'mhas': mhas}
		dirs.append(new_dir)
	return dirs

def crawl(path, item, mhas):
	if item.endswith('.mha'):
		mha_path = os.path.join(path, item)
		mhas.append(mha_path)
		return mhas
	else:
		new_path = os.path.join(path, item)
		if not os.path.isdir(new_path):
			return mhas
		for it in os.listdir(new_path):
			mhas = crawl(new_path, it, mhas)
		return mhas

def find_patients(dir_name):
	pats = []
	pats = get_patient_dirs(dir_name, pats)

	for p in pats:
		print p
	print 'Found %i patient directories.' % len(pats)

if __name__ == '__main__':
	if len(sys.argv) != 2:
		raise ValueError('You have to input the directory to be crawled.')
	dir_name = sys.argv[1]

	find_patients(dir_name)
