#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import numpy as np
import pdb
from pcseg.data.dataset.ceph import PetrelBackend
import multiprocessing

# possible splits
splits = ["train", "valid", "trainval", "test"]

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./remap_semantic_labels.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=False,
      default=None,
      help='Dataset dir. WARNING: This file remaps the labels in place, so the original labels will be lost. Cannot be used together with -predictions- flag.'
  )
  parser.add_argument(
      '--predictions', '-p',
      type=str,
      required=False,
      default=None,
      help='Prediction dir. WARNING: This file remaps the predictions in place, so the original predictions will be lost. Cannot be used together with -dataset- flag.'
  )
  parser.add_argument(
      '--split', '-s',
      type=str,
      required=False,
      default="valid",
      help='Split to evaluate on. One of ' +
      str(splits) + '. Defaults to %(default)s',
  )
  parser.add_argument(
      '--datacfg', '-dc',
      type=str,
      required=False,
      default="semantic-kitti-all.yaml",
      help='Dataset config file. Defaults to %(default)s',
  )
  parser.add_argument(
      '--inverse',
      dest='inverse',
      default=False,
      action='store_true',
      help='Map from xentropy to original, instead of original to xentropy. '
      'Defaults to %(default)s',
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Data: ", FLAGS.dataset)
  print("Predictions: ", FLAGS.predictions)
  print("Split: ", FLAGS.split)
  print("Config: ", FLAGS.datacfg)
  print("Inverse: ", FLAGS.inverse)
  print("*" * 80)

  # only predictions or dataset can be handled at once and one MUST be given (xor)
  assert((FLAGS.dataset is not None) != (FLAGS.predictions is not None))

  # check name
  root_directory = ""
  label_directory = ""
  if(FLAGS.dataset is not None):
    root_directory = FLAGS.dataset
    label_directory = "labels"
  elif(FLAGS.predictions is not None):
    root_directory = FLAGS.predictions
    label_directory = "predictions"
  else:
    print("I don't even know how I got here")
    quit()

  # assert split
  assert(FLAGS.split in splits)

  print("Opening data config file %s" % FLAGS.datacfg)
  DATA = yaml.safe_load(open(FLAGS.datacfg, 'r'))

  # get number of interest classes, and the label mappings
  if FLAGS.inverse:
    print("Mapping xentropy to original labels")
    remapdict = DATA["learning_map_inv"]
  else:
    remapdict = DATA["learning_map"]
  nr_classes = len(remapdict)

  # make lookup table for mapping
  maxkey = max(remapdict.keys())

  # +100 hack making lut bigger just in case there are unknown labels
  remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
  remap_lut[list(remapdict.keys())] = list(remapdict.values())
  # print(remap_lut)

  # get wanted set
  sequences = []
  sequences.extend(DATA["split"][FLAGS.split])

  if 's3://' in root_directory:
    petrel_client = PetrelBackend()
    for sequence in sequences:
      sequence = '{0:02d}'.format(int(sequence))
      label_dir = os.path.join(root_directory, "sequences",
                                sequence, label_directory)
      seq_label_names = petrel_client.list_dir_one_depth(label_dir)
      seq_label_names = [seq_label_name for seq_label_name in seq_label_names if seq_label_name.endswith('.label')]
      seq_label_names.sort()
      remap_dir = os.path.join(root_directory+'_remap', "sequences",
                                sequence, label_directory)

      def remap_single_frame(label_name):
        label_file = os.path.join(label_dir, label_name)
        print(label_file)
        label   = np.copy(petrel_client.load_bin(label_file, dtype='uint32').reshape((-1)))
        label = label.reshape((-1))
        upper_half = label >> 16      # get upper half for instances
        lower_half = label & 0xFFFF   # get lower half for semantics
        lower_half = remap_lut[lower_half]  # do the remapping of semantics
        label = (upper_half << 16) + lower_half   # reconstruct full label
        label = label.astype(np.uint32)
        remap_file = os.path.join(remap_dir, label_name)
        petrel_client.save_bin(remap_file, label)
        return remap_file
        
      with multiprocessing.Pool(32) as p:
          remap_path_list = list(p.map(remap_single_frame, seq_label_names))

  else:
    label_names = []
    for sequence in sequences:
      sequence = '{0:02d}'.format(int(sequence))
      label_paths = os.path.join(root_directory, "sequences",
                                sequence, label_directory)
      seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(label_paths)) for f in fn if ".label" in f]
      seq_label_names.sort()
      label_names.extend(seq_label_names)
    for label_file in label_names:
      print(label_file)
      label = np.fromfile(label_file, dtype=np.uint32)
      label = label.reshape((-1))
      upper_half = label >> 16      # get upper half for instances
      lower_half = label & 0xFFFF   # get lower half for semantics
      lower_half = remap_lut[lower_half]  # do the remapping of semantics
      label = (upper_half << 16) + lower_half   # reconstruct full label
      label = label.astype(np.uint32)
      label.tofile(label_file)