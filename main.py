# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from codes.ply_to_tsplvq import ply_to_tsplvq
from codes.start_Z import CompressSrcToPLY
import glob
import os
from personal_parameters.myConfig import config
from codes import NNdataProcess

# Press the green button in the gutter to run the script.

def Create_Tree():
    src_folder = config.pathToLongdressPly
    dest_folder = config.pathToLongdressCompressed_Ply
    method = "hybrid"  # or "2" or "3"
    rateMax = 10000
    verbose = True


    files = glob.glob(os.path.join(src_folder, "*.ply"))
    if verbose:
        print(len(files), "files to treat")
    for i, file in enumerate(files):
        filename = os.path.splitext(os.path.basename(file))[0]
        destination = os.path.normpath(os.path.join(dest_folder, filename + ".obj"))
        if verbose:
            print((i + 1), "/", len(files), ":", round((i / len(files)) * 100), "%")

        try:
            os.makedirs(dest_folder, exist_ok=True)
        except:
            pass
        print(file)
        #ply_to_tsplvq(file, destination, method, rateMax, verbose=True)
        CompressSrcToPLY(file ,src_folder, dest_folder, 10000)


    if verbose:
        print(len(files), "/", len(files), ": 100%\nDone.")

def Create_Dataset():

    raw_ply = config.pathToLongdress2Ply
    lable_dir = config.pathToLongdress2Compressed_Ply
    save_dir = config.pathToLongdress2RNNTrain
    NNdataProcess.create_training_sequence_list_wise_for_RNN(raw_ply, lable_dir, save_dir)





if __name__ == '__main__':
    import argparse

    #Create_Tree()
    Create_Dataset()