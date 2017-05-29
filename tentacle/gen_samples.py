import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from easydict import EasyDict as edict
from annotate import load_json, get_category
from PIL import Image
from skimage import img_as_float
from skimage import img_as_ubyte


cfg = edict()


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def record_cats(cats_json, cats_dirs, tfrecords_filename):
    cats = load_json(cats_json)
    cats = {k: v['name'] for k, v in cats.items()}
    class_to_catid = dict([(v, k) for k, v in cats.items()])

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


def rand_pos(bg_file, width, height, w1, h1, num):
    sess = tf.Session()

    tfrecords_filename = os.path.join(cfg.dat_dir, 'small.tfrecords')
    smalls = load_record(tfrecords_filename)
    tmp = []
    for img, catid in smalls:
        img_op = tf.image.convert_image_dtype(img, tf.float32)
        img = sess.run(img_op)
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

    n_blocks = (width//w1) * (height//h1)

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

    print(norm_bboxes.shape)
    img = np.array(Image.open(bg_file).convert('RGB'))
    img = np.asanyarray(img_as_float(img), np.float32)
    img_color_op = distort_color(img)
    img = sess.run(img_color_op)
    print(img.shape, img.dtype)
    fig, ax = plt.subplots(1)

    idx = np.random.choice(norm_bboxes.shape[0], size=num, replace=False)
    for y, x, y1, x1 in norm_bboxes[idx, :]:
        w, h = x1-x, y1-y
#         print(x, y, x1, y1, w, h)
#             rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=np.random.rand(3, 1), facecolor='none')
#             ax.add_patch(rect)
        small_idx = np.random.choice(len(smalls), size=1)[0]
        small_img, small_catid = smalls[small_idx][0], smalls[small_idx][1]
        rot = np.random.randint(4, size=1)[0]
        new_small = distort_image(small_img, h, w, [[[0, 0., 1, 1]]], degree_rotate=rot)

        small1 = sess.run(new_small)
        print(small1.shape, small1.dtype)
        img[y:y+h, x:x+w, :] = small1
    ax.imshow(img)
    plt.show()

    sess.close()


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
    image_rop = distort_image(image, 600, 800, tf.constant([[[1 / 6, 1 / 9, 2 / 6, 2 / 9],
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

def distort_image(image, height, width, bbox, thread_id=0, scope=None, degree_rotate=0):
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
  with tf.name_scope(values=[image, height, width, bbox], name=scope,
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
    distorted_image = tf.image.resize_images(distorted_image, [height, width],
                                             method=resize_method)

#     distorted_image = tf.image.rot90(distorted_image, k=degree_rotate)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([height, width, 3])
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
    rand_pos(bg_file, 1024, 1024, 128, 128, 14)
