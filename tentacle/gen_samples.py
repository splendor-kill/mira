import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
from easydict import EasyDict as edict

cfg = edict()

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
    image_rop = distort_image(image, 600, 800, tf.constant([[[1 / 6, 1 / 9, 2 / 6, 2 / 9]]]))
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
        # Write summary

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

def distort_image(image, height, width, bbox, thread_id=0, scope=None):
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
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
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
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([height, width, 3])
    if not thread_id:
      tf.summary.image('cropped_resized_image',
                       tf.expand_dims(distorted_image, 0))
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # Randomly distort the colors.
    distorted_image = distort_color(distorted_image, thread_id)
    if not thread_id:
      tf.summary.image('final_distorted_image',
                       tf.expand_dims(distorted_image, 0))
    return distorted_image


if __name__ == '__main__':
    cfg.dat_dir = '/home/splendor/jumbo/annlab/mj/'
    bg_file = os.path.join(cfg.dat_dir, 'mjatlas.png')
    gen(bg_file)
