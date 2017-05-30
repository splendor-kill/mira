import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import glob
from easydict import EasyDict as edict
from annotate import load_json, get_category
from PIL import Image
from skimage import img_as_float
from skimage import img_as_ubyte
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement
import time

cfg = edict()


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_cats(cats_json):
    cats = load_json(cats_json)
    cats = {k: v['name'] for k, v in cats.items()}
    class_to_catid = dict([(v, k) for k, v in cats.items()])
    return cats, class_to_catid


def record_cats(cats_json, cats_dirs, tfrecords_filename):
    _, class_to_catid = load_cats(cats_json)

    records = []
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    for cats_dir in cats_dirs:
        for root, _, files in os.walk(cats_dir):
            for file_name in files:
                catg = get_category(file_name)
                full_file_name = os.path.join(root, file_name)
                img = np.array(Image.open(full_file_name).convert('RGB'))
                catid = class_to_catid[catg]
                records.append((img, catid))
#                 print(img.shape, img.dtype)
                example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(img.shape[0]),
                'width': _int64_feature(img.shape[1]),
                'depth': _int64_feature(img.shape[2]),
                'catg': _int64_feature(catid),
                'image': _bytes_feature(img.tostring())}))
                writer.write(example.SerializeToString())
    writer.close()
    return records


def load_record(tfrecords_filename):
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    reconstructed_images = []
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        depth = int(example.features.feature['depth'].int64_list.value[0])
        catg = int(example.features.feature['catg'].int64_list.value[0])
        img_string = (example.features.feature['image'].bytes_list.value[0])
        img_1d = np.fromstring(img_string, dtype=np.uint8)
        reconstructed_img = img_1d.reshape((height, width, -1))
        reconstructed_images.append((reconstructed_img, catg))
    return reconstructed_images


def save_to_xml(folder, width, height, bbs, cats, new_file_name, file):
    root = Element('annotation')
    SubElement(root, 'folder').text = folder
    SubElement(root, 'filename').text = new_file_name
    size_node = SubElement(root, 'size')
    SubElement(size_node, 'width').text = str(width)
    SubElement(size_node, 'height').text = str(height)

    for catg, x, y, x1, y1 in bbs:
        object_node = SubElement(root, 'object')
        SubElement(object_node, 'name').text = cats[catg]
        bndbox_node = SubElement(object_node, 'bndbox')
        SubElement(bndbox_node, 'xmin').text = str(x)
        SubElement(bndbox_node, 'ymin').text = str(y)
        SubElement(bndbox_node, 'xmax').text = str(x1)
        SubElement(bndbox_node, 'ymax').text = str(y1)

    tree = ET.ElementTree(root)
    tree.write(file)


def rand_pos(sess, sample_distorted_bounding_box, bg_rand_op,
             distort_op, img_float_pl, out_size_pl, times_rot90_pl,
             smalls, bg_img, width, height, w1, h1, num):
    n_blocks = (width // w1) * (height // h1)

    distorted_bboxes = []
    for i in range(n_blocks):
        bbox = sess.run(sample_distorted_bounding_box)
        begin, size, _ = bbox
        distorted_bboxes.append([begin[0], begin[1], size[0], size[1]])
    distorted_bboxes = np.array(distorted_bboxes)

    norm_bboxes = []
    cols = width // w1
    for i, row in enumerate(distorted_bboxes):
        loc = divmod(i, cols)
        offset = (loc[0] * h1, loc[1] * w1)
        lt = (offset[0] + row[0], offset[1] + row[1])
        br = (lt[0] + row[2], lt[1] + row[3])
        norm_bboxes.append((*lt, *br))
    norm_bboxes = np.array(norm_bboxes)

    img = sess.run(bg_rand_op, feed_dict={img_float_pl: bg_img})
#     print(img.shape, img.dtype)
#     fig, ax = plt.subplots(1)

    bbs = []
    idx = np.random.choice(norm_bboxes.shape[0], size=num, replace=False)
    for y, x, y1, x1 in norm_bboxes[idx, :]:
        w, h = x1 - x, y1 - y
#         print(x, y, x1, y1, w, h)
#             rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=np.random.rand(3, 1), facecolor='none')
#             ax.add_patch(rect)
        small_idx = np.random.choice(len(smalls), size=1)[0]
        small_img, small_catid = smalls[small_idx][0], smalls[small_idx][1]
        rot = np.random.randint(4, size=1)[0]
#         new_small = distort_image(small_img, h, w, [[[0, 0., 1, 1]]], times_rot90=rot)

        small1 = sess.run(distort_op,
                          feed_dict={img_float_pl: small_img,
                                     out_size_pl: (h, w),
                                     times_rot90_pl: rot})

#         print(small1.shape, small1.dtype)
        if h == small1.shape[1]:
            img[y:y + h, x:x + w, :] = np.fliplr(small1.swapaxes(0, 1))
        else:
            img[y:y + h, x:x + w, :] = small1

        bbs.append((small_catid, x, y, x1, y1))
#     ax.imshow(img)
#     plt.show()

    return img, bbs


def augment(bg_file, width, height, w1, h1, num_pieces, num_imgs, image_dir, prefix, annos_dir, cats_json):
    cats, _ = load_cats(cats_json)
    next_image_id = get_max_image_id(prefix, annos_dir) + 1

    tfrecords_filename = os.path.join(cfg.dat_dir, 'small.tfrecords')
    smalls = load_record(tfrecords_filename)

    bg_img = np.array(Image.open(bg_file).convert('RGB'))
    bg_img = np.asanyarray(img_as_float(bg_img), np.float32)

    sess = tf.Session()

    img_pl = tf.placeholder(tf.uint8, [None, None, 3], name='u8_img')
    img_op = tf.image.convert_image_dtype(img_pl, tf.float32)
    tmp = []
    for img, catid in smalls:
        img = sess.run(img_op, feed_dict={img_pl: img})
        tmp.append((img, catid))
    smalls = tmp

    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        [w1, h1, 3],
        bounding_boxes=[[[0, 0., 1, 1]]],
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.75, .9],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)

    img_float_pl = tf.placeholder(tf.float32, shape=[None, None, 3], name='fp_img')
    out_size_pl = tf.placeholder(tf.int32, shape=[2], name='out_size')
    times_rot90_pl = tf.placeholder(tf.int32, shape=[], name='times_rot90')
    distort_op = distort_image(img_float_pl, out_size_pl, [[[0, 0., 1, 1]]], times_rot90=times_rot90_pl)
    bg_rand_op = distort_color(img_float_pl, scope='bg_rand_color')

    writer = tf.summary.FileWriter('../logs', graph=sess.graph)

    begin = time.time()
    costs = []
    for i in range(num_imgs):
        t1 = time.time()
        img, bbs = rand_pos(sess, sample_distorted_bounding_box, bg_rand_op,
                            distort_op, img_float_pl, out_size_pl, times_rot90_pl,
                            smalls, bg_img, width, height, w1, h1, num_pieces)
        t2 = time.time()
        new_file_name = '%s_%05d' % (prefix, next_image_id)
        img = Image.fromarray(img_as_ubyte(img))
        img.save(os.path.join(image_dir, new_file_name + '.png'))
        save_to_xml(os.path.relpath(image_dir, cfg.dat_dir), width, height, bbs, cats,
                    new_file_name + '.png', os.path.join(annos_dir, new_file_name + '.xml'))
        t3 = time.time()
        next_image_id += 1
        costs.append((t2-t1, t3-t2))
        if i % 100 == 0:
            print('%-5d: %.3f' % (i, sum(map(sum, zip(*costs))) / len(costs)))

    costs = np.array(costs)
    costs_avg = costs.mean(axis=0)
    total = time.time() - begin
    print('time cost(s): all(%.3f), avg(%.3f), gen_avg(%.3f), save_avg(%.3f)' % (total, total / num_imgs, costs_avg[0], costs_avg[1]))
    writer.close()
    sess.close()


def get_max_image_id(prefix, annos_dir):
    max_id = 0
    for f in glob.glob(os.path.join(annos_dir, '%s_*.xml' % (prefix,))):
        f = os.path.basename(f)
        f = os.path.splitext(f)[0]
        f = f.replace('%s_' % (prefix,), '')
        max_id = max(max_id, int(f))
    return max_id


def gen(bg_file):
    '''generate image and its annotations'''
    image_reader = tf.WholeFileReader()

    filename_queue = tf.train.string_input_producer([bg_file])

    _, value = image_reader.read(filename_queue)

    image = tf.image.decode_png(value, 3)
    image = tf.image.convert_image_dtype(image, tf.float32)
#    image_bb = tf.image.draw_bounding_boxes(
#            tf.expand_dims(image, 0),
#            tf.constant([[[0.1, 0.2, 0.5, 0.9]]]))
#    image_rop = tf.image.random_brightness(image, 0.3)
#    image_rop = tf.image.random_hue(image, 0.3)
#    image_rop = tf.image.random_contrast(image, 0.1, 0.9)
#    image_rop = tf.image.random_saturation(image, 0.1, 0.9)
#    image_rop = tf.image.per_image_standardization(image)

    image_rop = distort_image(image, (600, 800), tf.constant([[[1 / 6, 1 / 9, 2 / 6, 2 / 9],
                                                             [2 / 6, 4 / 9, 3 / 6, 5 / 9],
                                                             [3 / 6, 5 / 9, 4 / 6, 6 / 9]]]))
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    summary_op = tf.summary.merge(summaries)
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        writer = tf.summary.FileWriter('../logs', graph=sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        img, img_rop = sess.run([image, image_rop])
        print(img_rop.shape, img.dtype)

#        plt.imshow(img_bb[0,:,:,:])


        summary = sess.run(summary_op)
        writer.add_summary(summary)


        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(img)
        fig.add_subplot(1, 2, 2)
        plt.imshow(img_rop)
        plt.show()

        coord.request_stop()
        coord.join(threads)

        writer.close()

def distort_color(image, thread_id=0, scope=None):
  """Distort the color of the image.
  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.
  Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for name_scope.
  Returns:
    color-distorted image
  """
  with tf.name_scope(values=[image], name=scope, default_name='distort_color'):
    color_ordering = thread_id % 2
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def distort_image(image, size, bbox, thread_id=0, scope=None, times_rot90=0):
  """Distort one image for training a network.
  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.
  Args:
    image: 3-D float Tensor of image
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    thread_id: integer indicating the preprocessing thread.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of distorted image used for training.
  """
  with tf.name_scope(values=[image, size, bbox, times_rot90], name=scope,
                     default_name='distort_image'):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].
    # Display the bounding box in the first thread only.
    if not thread_id:
      image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                    bbox)
      tf.summary.image('image_with_bounding_boxes', image_with_box)
  # A large fraction of image datasets contain a human-annotated bounding
  # box delineating the region of the image containing the object of interest.
  # We choose to create a new bounding box for the object which is a randomly
  # distorted version of the human-annotated bounding box that obeys an allowed
  # range of aspect ratios, sizes and overlap with the human-annotated
  # bounding box. If no box is supplied, then we assume the bounding box is
  # the entire image.
    bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=0.5,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.90, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)

#     print(distort_bbox.get_shape())

    if not thread_id:
      image_with_distorted_box = tf.image.draw_bounding_boxes(
          tf.expand_dims(image, 0), distort_bbox)
      tf.summary.image('images_with_distorted_bounding_box',
                       image_with_distorted_box)
    # Crop the image to the specified bounding box.
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    # This resizing operation may distort the images because the aspect
    # ratio is not respected. We select a resize method in a round robin
    # fashion based on the thread number.
    # Note that ResizeMethod contains 4 enumerated resizing methods.
    resize_method = thread_id % 4
    distorted_image = tf.image.resize_images(distorted_image, size,
                                             method=resize_method)

#     distorted_image.set_shape([height, width, 3])
    if not thread_id:
      tf.summary.image('cropped_resized_image',
                       tf.expand_dims(distorted_image, 0))
    # Randomly flip the image horizontally.
#     distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Randomly distort the colors.
    distorted_image = distort_color(distorted_image, thread_id)
    if not thread_id:
      tf.summary.image('final_distorted_image',
                       tf.expand_dims(distorted_image, 0))

    distorted_image = tf.image.rot90(distorted_image, k=times_rot90)

    return distorted_image


if __name__ == '__main__':
    cfg.dat_dir = '/home/splendor/win/mira/dataset/'
    bg_file = os.path.join(cfg.dat_dir, 'mj', 'bg_1.png')
#     gen(bg_file)
#     cats_dirs = ['atomitems', 'tianfeng_small', 'uni_small', 'wiki_small']
#     cats_dirs = [os.path.join(cfg.dat_dir, 'mj', d) for d in cats_dirs]
#     print(cats_dirs)
#     record_cats(os.path.join(cfg.dat_dir, 'cats.json'),
#                 cats_dirs,
#                 os.path.join(cfg.dat_dir, 'small.tfrecords'))
#     images = load_record(os.path.join(cfg.dat_dir, 'small.tfrecords'))

    augment(bg_file, 1024, 1024, 128, 128, 14, 10,
            cfg.dat_dir + 'mj/ds_gen/images',
            'generated',
            cfg.dat_dir + 'mj/ds_gen/annotations',
            cfg.dat_dir + 'cats.json')
