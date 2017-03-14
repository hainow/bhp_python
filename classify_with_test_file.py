# THIS FILE WILL BE CALLED BY "batch_test.sh"

#!/usr/bin/env python2
# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.

"""
Classify an image using individual model files

Use this script as an example to build your own tool
"""

import argparse
import os
import time

from google.protobuf import text_format
import numpy as np
import PIL.Image
import scipy.misc

os.environ['GLOG_minloglevel'] = '2' # Suppress most caffe output
import caffe
from caffe.proto import caffe_pb2


def get_net(caffemodel, deploy_file, use_gpu=True):
    """
    Returns an instance of caffe.Net

    Arguments:
    caffemodel -- path to a .caffemodel file
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    use_gpu -- if True, use the GPU for inference
    """
    if use_gpu:
        caffe.set_mode_gpu()

    # load a new model
    return caffe.Net(deploy_file, caffemodel, caffe.TEST)


def get_transformer(deploy_file, mean_file=None):
    """
    Returns an instance of caffe.io.Transformer

    Arguments:
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    mean_file -- path to a .binaryproto file (optional)
    """
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    if network.input_shape:
        dims = network.input_shape[0].dim
    else:
        dims = network.input_dim[:4]

    t = caffe.io.Transformer(
        inputs={'data': dims}
    )
    t.set_transpose('data', (2, 0, 1))  # transpose to (channels, height, width)

    # color images
    if dims[1] == 3:
        # channel swap
        t.set_channel_swap('data', (2, 1, 0))

    if mean_file:
        # set mean pixel
        with open(mean_file, 'rb') as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            t.set_mean('data', pixel)

    return t

def load_image(path, height, width, mode='RGB'):
    """
    Load an image from disk

    Returns an np.ndarray (channels x width x height)

    Arguments:
    path -- path to an image on disk
    width -- resize dimension
    height -- resize dimension

    Keyword arguments:
    mode -- the PIL mode that the image should be converted to
        (RGB for color or L for grayscale)
    """
    image = PIL.Image.open(path)
    image = image.convert(mode)
    image = np.array(image)
    # squash
    image = scipy.misc.imresize(image, (height, width), 'bilinear')
    return image

def forward_pass(images, net, transformer, batch_size=256):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)

    Arguments:
    images -- a list of np.ndarrays
    net -- a caffe.Net
    transformer -- a caffe.io.Transformer

    Keyword arguments:
    batch_size -- how many images can be processed at once
        (a high value may result in out-of-memory errors)
    """
    if batch_size is None:
        batch_size = 1

    caffe_images = []
    for image in images:
        if image.ndim == 2:
            caffe_images.append(image[:,:,np.newaxis])
        else:
            caffe_images.append(image)

    dims = transformer.inputs['data'][1:]

    scores = None
    for chunk in [caffe_images[x:x+batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        start = time.time()
        output = net.forward()[net.outputs[-1]]
        end = time.time()
        if scores is None:
            scores = np.copy(output)
        else:
            scores = np.vstack((scores, output))
        #print 'Processed %s/%s images in %f seconds ...' % (len(scores), len(caffe_images), (end - start))

    return scores

def read_labels(labels_file):
    """
    Returns a list of strings

    Arguments:
    labels_file -- path to a .txt file
    """
    if not labels_file:
        print 'WARNING: No labels file provided. Results will be difficult to interpret.'
        return None

    labels = []
    with open(labels_file) as infile:
        for line in infile:
            label = line.strip()
            if label:
                labels.append(label)
    assert len(labels), 'No labels found'
    return labels

def label_to_idx(labels):
    l_to_i = {}
    for idx, label in enumerate(labels):
        l_to_i[label] = idx
    return l_to_i

def get_img_paths_and_labels(test_file):
    test_paths, test_indices = [], []
    with open(test_file) as f:
        for line in f:
            fields = line.strip().split()
            # process path
            path = fields[0]
            assert(os.path.isfile(path))
            test_paths.append(path)
            # and update label index
            test_indices.append(int(fields[1]))

    return test_paths, test_indices

# def get_img_paths_and_labels(test_file):
#     ''' generator version'''
#     test_paths, test_indices = [], []
#     with open(test_file) as f:
#         for line in f:
#             fields = line.strip().split()
#             # process path
#             path = fields[0]
#             assert(os.path.isfile(path))
#             yield (path, int(fields[1]))


def classify(caffemodel, deploy_file, test_file,
        mean_file=None, labels_file=None, batch_size=256, use_gpu=True):
    """
    Classify some images against a Caffe model and print the results

    Arguments:
    caffemodel -- path to a .caffemodel
    deploy_file -- path to a .prototxt
    test_file -- list of paths to images and associated labels

    Keyword arguments:
    mean_file -- path to a .binaryproto
    labels_file path to a .txt file
    use_gpu -- if True, run inference on the GPU
    """
    # Load the model and images
    net = get_net(caffemodel, deploy_file, use_gpu)
    transformer = get_transformer(deploy_file, mean_file)
    _, channels, height, width = transformer.inputs['data']
    if channels == 3:
        mode = 'RGB'
    elif channels == 1:
        mode = 'L'
    else:
        raise ValueError('Invalid number for channels: %s' % channels)


    labels = read_labels(labels_file)
    test_paths, test_indices = get_img_paths_and_labels(test_file)
    # get 2 lists which are associated to each other
    test_images = [load_image(image_file, height, width, mode) for image_file in test_paths]
    test_labels = [labels[i] for i in test_indices]
    test_len = len(test_labels)

    print "test label indices"
    print test_indices

    # Classify the image
    scores = forward_pass(test_images, net, transformer, batch_size=batch_size)

    ### Process the results

    indices = (-scores).argsort()[:, :5] # take top 5 results
    # indices = (-scores).argsort()[:, :1] # take top 1 results ONLY

    classifications = []
    top1, top5 = 0, 0
    count = 1
    for image_index, index_list in enumerate(indices): # e.g. 0 [13 36  5 25 34]
        # print image_index, index_list
        result = []
        for i in index_list:
            # 'i' is a category in labels and also an index into scores
            if labels is None:
                label = 'Class #%s' % i
            else:
                label = labels[i]
            result.append((label, round(100.0*scores[image_index, i],4)))
        classifications.append(result)

        # populate top 1
        if index_list[0] == test_indices[image_index]:
            top1 += 1
        # populate top 5
        if test_indices[image_index] in index_list:
            top5 += 1

        count += 1

        if count % 1000 == 0:
            print('At sample of {}'.format(count))
            print('Current top1 = {}, acc = {}'.format(top1, float(top1)/count * 100))
            print('Current top5 = {}, acc = {}'.format(top5, float(top5) / count * 100))

    print("\n\nSUMMARY:\n----------\n Top 1 = {}\t(acc={} %) \n Top 5 = {}\t(acc={} %)".
          format(top1, float(top1)/test_len * 100, top5, float(top5)/test_len * 100))
    print("\nResult is written into file: result.txt\n\n")

    # write to result.txt
    f = open('result.txt', 'w')
    for index, classification in enumerate(classifications):
        # print '{:-^80}'.format(' Prediction for %s ' % test_paths[index])
        # print '{:-^80}'.format('(label = %s)' % test_labels[index])
        f.write(test_paths[index] + ', label: ' + test_labels[index] + '\n')
        f.write('-------------------------------------------------\n')
        for label, confidence in classification:
            # print '{:9.4%}\t{}'.format(confidence/100.0, label)
            f.write('\t' + label + '\t' + str(confidence) + '%\n')
        f.write('\n\n')
        # print
    f.close()

def main():
    script_start_time = time.time()

    parser = argparse.ArgumentParser(description='Classification example - DIGITS')

    ### Positional arguments

    parser.add_argument('caffemodel',   help='Path to a .caffemodel')
    parser.add_argument('deploy_file',  help='Path to the deploy file')
    # parser.add_argument('image_file',
    #                     nargs='+',
    #                     help='Path[s] to an image')
    parser.add_argument('-t', '--testfile', help='a text file specifying paths to testing images and associated label indices')

    ### Optional arguments

    parser.add_argument('-m', '--mean',
            help='Path to a mean file (*.npy)')
    parser.add_argument('-l', '--labels',
            help='Path to a labels file')
    parser.add_argument('--batch-size',
                        type=int)
    parser.add_argument('--nogpu',
            action='store_true',
            help="Don't use the GPU")

    args = vars(parser.parse_args())

    classify(args['caffemodel'], args['deploy_file'], args['testfile'],
            args['mean'], args['labels'], args['batch_size'], not args['nogpu'])

    print 'Script\ttook %f seconds.' % (time.time() - script_start_time,)


def test():
    labels = read_labels('./testmdb/labels.txt')
    print labels
    l2i = label_to_idx(labels)
    print l2i
    print labels[0]

    paths, indices = get_img_paths_and_labels('./testmdb/test.txt')
    print paths
    print indices
    for i in indices: print labels[i],

    # generator test
    # for path, idx in get_img_paths_and_labels('./testmdb/test.txt'):
    #     print path, idx



if __name__ == '__main__':
    # main()
    test()
