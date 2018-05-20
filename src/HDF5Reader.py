import h5py
from array import array
class HDF5Reader:
	def __init__(self, file):
		self.file  = file
		self.docVectors = {} # text to list of its vector ("a new bedford..." -> [[][][])
		f = h5py.File(self.file, 'r')
		self.extract_datasets(f)
	def extract_datasets(self, items):
		for key in items.keys():
			item = items[key]
			#print("key:",key)
			#exit(1)
			if isinstance(item, h5py.Dataset):  # test for dataset
				self.docVectors[key.lower()] = item.value

			elif isinstance(item, h5py.Group):  # test for group (go down)
				print("key was w/ group:",key)
				self.extract_datasets(item)
