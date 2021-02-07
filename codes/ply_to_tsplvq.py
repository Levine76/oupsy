# need :
# pip install argparse
# pip install pickle

import glob
import os
import pickle
from codes import ObjLoader
from codes import TrainingSequence
#from codes.TrainingSequence import TrainingSequence
from codes import Quantization
#from codes.Quantization import Quantization
from codes import Tree
from codes import QuantizationMethodBase
#from codes.QuantizationMethodBase import QuantizationMethodBase
import sys
from codes import Probability
from personal_parameters.myConfig import config

# CMD : python ply_to_tsplvq.py -src="dossier/avec/les/PLY/" -dest="dossier/pour/enregistrer" -methods="2" -rate="30000" -v

###
# This script is used to transform a .PLY file into a TSPLVQ object
# file is the .PLY file to transform
# dest is the desired output file. Use None if you don't want to store it
# methods are the methods used (2 = 2x2x2, 3 = 3x3x3, hybrid = [2 and 3])
# rateMax is the maximal rate of the tree
# verbose for output
###

# TODO
# This method should be implemented in Tree.loadFromPly()
def ply_to_tsplvq(file, dest, methods=[2, 3], rateMax=10000, verbose=False):
	if not isinstance(methods, list) :
		try:
			methods = [int(methods)] # In case it's the string "2" or "3"
		except:
			methods = [2, 3] # In case it's the string "hybrid"

	## loading points cloud
	pcd = ObjLoader.load_point_cloud(file)
	print("taille du point cloud chargé : ", pcd.__sizeof__())
	## Creating training sequence
	ts = TrainingSequence.TrainingSequence()
	ts.init(pcd)
	ts.compute_normals()
	## Creating quantization instance
	qt = Quantization.Quantization()
	for method in methods:
		qt.add_method(QuantizationMethodBase.QuantizationMethodBase(method))
	## Creating tree
	tree = Tree.Tree(ts)
	tree.create_root()

	# Entropic code
	pro = Probability.Probability()

	## Quantization Loop
	depthLimit = 20
	while int(tree.rate) < int(rateMax):
		qt.quantize(tree, pro)
		if tree.maxDepth() >= depthLimit:
			break
	pro.indexing_states()

	if dest is not None:
		# sauvegarde du Tree en objet pickle
		file = open(dest,'wb')
		pickle.dump(tree, file)
		file.close()

	return tree







if __name__ == "__main__":
	import argparse
	src_folder = "../data/soldier/Ply/" #config.pathToLongdressPly
	dest_folder = "../data/soldier/Compressed_Ply/"#config.pathToLongdressTSPLVQ
	method = "hybrid"  # or "2" or "3"
	rateMax = 10000
	verbose = True


	# Create the "help" with argparse
	parser=argparse.ArgumentParser(
		description='''Transformation d'une sequence de PLY en objet Tree ''')
	parser.add_argument('-src', default=src_folder, 
		help='Dossier contenant les fichiers PLY')
	parser.add_argument('-destination', '-dest', default=dest_folder, 
		help='Dossier ou deposer les arbres TSPLVQ')
	parser.add_argument('-method', default=method, const=method, choices=["2", "3", "hybrid"], nargs="?",
		help='Methode utilisee pour realiser l\'arbre TSPLVQ ("2" = 2x2x2, "3" = 3x3x3, "hybrid" = melange)')
	parser.add_argument('-rate', default=rateMax, 
		help='Nombre maximal de feuilles dans l\'arbre')
	parser.add_argument('-verbose', '-v',  action="store_true",
		help='Mode verbal')
	args = vars(parser.parse_args())

	for opt in args:
		if opt == "src" :
			src_folder = args[opt] 
		elif opt in ("dest", "destination") :
			dest_folder = args[opt] 
		elif opt == "method" :
			method = args[opt]
		elif opt == "rate" :
			rateMax = args[opt]

	files = glob.glob(os.path.join(src_folder, "*.ply"))
	if verbose :
		print(len(files), "files to treat")
	for i, file in enumerate(files):
		filename = os.path.splitext(os.path.basename(file))[0]
		destination = os.path.normpath(os.path.join(dest_folder, filename + ".obj"))
		if verbose:
			print((i+1), "/", len(files), ":", round((i/len(files))*100), "%")

		try :
			os.makedirs(dest_folder, exist_ok=True)
		except:
			pass
		ply_to_tsplvq(file, destination, method, rateMax, verbose=True)

	if verbose:
		print(len(files), "/", len(files), ": 100%\nDone.")
