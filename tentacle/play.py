import os
import numpy as np
import cv2
from easydict import EasyDict as edict

from annotate import load_json
from annotate import save_to_json

cfg = edict()
cfg.data_dir = '.'


def browse2(file_dir, template):
    for root, _, files in os.walk(file_dir):
        for file_name in files:
            full_file_name = os.path.join(root, file_name)
            image = cv2.imread(full_file_name)
            if image is None:
                print('file read error', full_file_name)
                continue
            detect2(image, template)
            # k = cv2.waitKey(0)
            # if k == ord('q'):
            #     return


def detect2(trainingImage, queryImage):
    # # sift = cv2.ORB_create()
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(queryImage, None)
    kp2, des2 = sift.detectAndCompute(trainingImage, None)

    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)  # or pass empty dictionary

    matcher = cv2.FlannBasedMatcher(indexParams, searchParams)

    # # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # # matcher = cv2.BFMatcher()

    # # matches = matcher.match(des1, des2)
    matches = matcher.knnMatch(des1, des2, k=2)

    # # matches = sorted(matches, key=lambda x: x.distance)
    # # resultImage = cv2.drawMatches(queryImage, kp1, trainingImage, kp2, matches[:10], None, flags=2)

    good = []
    # # matchesMask = [[0, 0] for i in range(len(matches))]

    # # David G. Lowe's ratio test, populate the mask
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            # matchesMask[i] = [1, 0]
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = queryImage.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        trainingImage = cv2.polylines(trainingImage, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    drawParams = dict(matchColor=(0, 255, 0),
                      singlePointColor=(255, 0, 0),
                      matchesMask=matchesMask,
                      flags=2)
    # resultImage = cv2.drawMatchesKnn(queryImage, kp1, trainingImage, kp2, matches, None, **drawParams)
    resultImage = cv2.drawMatches(queryImage, kp1, trainingImage, kp2, good, None, **drawParams)
    cv2.imshow('image', resultImage)
    cv2.waitKey(0)


def detect1(image, template):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    bbs = []
    for pt in zip(*loc[::-1]):
        bbs.append((pt[0], pt[1], w, h))
    return bbs


# template = cv2.imread('/home/splendor/win/mira/dataset/atomitems/t3.png', 0)
# assert template is not None
# image = cv2.imread('/home/splendor/win/mira/dataset/bigpics/sikuliximage-1493208890342.png')

# bbs = detect1(image, template)
# for x, y, w, h in bbs:
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# plt.imshow(image), plt.show()

# cv2.imshow('result', trainingImage)
# cv2.waitKey()


def approx_reduce(x, fluctation):
    x_inc = np.sort(x)
    si = np.argsort(x, kind='mergesort')
    delta = np.diff(x_inc)
    delta[delta < fluctation] = 0
    x_keep = np.ones_like(x_inc)
    x_keep[1:] = np.sign(delta)
    remains = x_inc[x_keep > 0]
    stairs = np.cumsum(x_keep) - 1
    reduced = np.array([remains[i] for i in stairs])
    return reduced[np.argsort(si)]


def reduce_bbs(bbs, fluctation):
    assert len(bbs) > 0
    a = np.array(bbs)
    print(a.shape, a.dtype)
    x = a[:, 0]
    y = a[:, 1]
    x_ = approx_reduce(x, fluctation)
    y_ = approx_reduce(y, fluctation)
    a[:, 0] = x_
    a[:, 1] = y_
    ua = np.unique(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))).view(a.dtype).reshape(-1, a.shape[1])
    return ua


def browse(file_dir, template):
    for root, _, files in os.walk(file_dir):
        for file_name in files:
            full_file_name = os.path.join(root, file_name)
            image = cv2.imread(full_file_name)
            if image is None:
                print('file read error', full_file_name)
                continue
            bbs = detect1(image, template)
            if len(bbs) == 0:
                print('not found')
            else:
                bbs = reduce_bbs(bbs, 2)
                for x, y, w, h in bbs:
                    print('bbox:', x, y, w, h)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print()
            x0, y0, w, h, gap = 286, 582, 56, 66, 16
            cv2.rectangle(image, (x0, y0), (x0 + 14 * w + gap, y0 + h), (255, 0, 0), 2)
            cv2.imshow('pic browser', image)
            k = cv2.waitKey(0)
            if k == ord('q'):
                return


def get_pieces(pic):
    name = os.path.basename(pic)
    name = os.path.splitext(name)[0]
    print(name)

    img = cv2.imread(pic)
    assert img is not None
    print(img.shape)

    hands = []
    x0, y0, w, h, gap, ml, mr = 286, 583, 56, 65, 16, 2, 1
    for i in range(13):
        x = x0 + w * i + ml
        s = img[y0:y0 + h, x:x + w - ml - mr, :]
        # print(s.shape)
        hands.append(s)

    x = x0 + 13 * w + gap
    s = img[y0:y0 + h, x:x + w, :]
    hands.append(s)

    # for i, s in enumerate(hands):
    #     print(i+1)
    #     cv2.imwrite('%s_%d.png' % (name, i+1), s)
    #     cv2.imshow('piece', s)
    #     cv2.waitKey()

    return hands


def get_pieces_batch(file_dir, out_dir):
    hands = []
    for root, _, files in os.walk(file_dir):
        for file_name in files:
            full_file_name = os.path.join(root, file_name)
            hands.extend(get_pieces(full_file_name))

    print('pieces num:', len(hands))
    for i, s in enumerate(hands):
        out_file = os.path.join(out_dir, '%s_%d.png' % ('p', i + 1))
        cv2.imwrite(out_file, s)


def hist_cats(cats):
    hists = {}
    for k, v in cats.items():
        full_name = os.path.join(cfg.data_dir, v['folder'], v['file_name'])
        # print(full_name)
        # img = cv2.imread(full_name, 0)
        img = cv2.imread(full_name)
        assert img is not None
        # h = cv2.calcHist([img], [0], None, [256], [0, 256])
        h = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(h, h)
        hists[k] = h.flatten()
    return hists


def most_like_hist(image, hists):
    # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # h = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    h = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(h, h)
    h_img = h.flatten()
    similarities = [(k, cv2.compareHist(h_img, h, cv2.HISTCMP_CORREL)) for k, h in hists.items()]
    similarities = np.array(similarities)
    # print('sims shape:', similarities.shape)
    idx_most_like = np.argmax(similarities[:, 1])
    return similarities[idx_most_like, 0]


def test_hist(file_dir):
    import json
    cats_file = os.path.join(cfg.data_dir, 'cats.json')
    print(cats_file)
    with open(cats_file) as f:
        cats = json.load(f)
    hists = hist_cats(cats)
    # print('len(hists) = %d' % len(hists))

    font = cv2.FONT_HERSHEY_SIMPLEX

    for root, _, files in os.walk(file_dir):
        for file_name in files:
            full_file_name = os.path.join(root, file_name)
            img = cv2.imread(full_file_name)
            assert img is not None

            # pieces = []
            x0, y0, w, h, gap, ml, mr = 285, 583, 56, 65, 16, 2, 1
            for i in range(14):
                x = x0 + w * i + ml
                x += gap if i == 13 else 0
                s = img[y0:y0 + h, x:x + w - ml - mr, :]
                cat_id = most_like_hist(s, hists)
                print('most like:', cats[cat_id]['name'])
                # pieces.append({'bbox': (x, y0, w-ml-mr, h), 'cat_id': cat_id})
                cv2.rectangle(img, (x, y0), (x + w - ml - mr, y0 + h), (255, 0, 0), 2)
                cv2.putText(img, cats[cat_id]['name'], (x, y0), font, 1, (255, 255, 0), 2)

            cv2.imshow('pic browser', img)
            k = cv2.waitKey(0)
            if k == ord('q'):
                return


def cats_feature(sift, cats):
    dess = {}
    for k, v in cats.items():
        full_name = os.path.join(cfg.data_dir, v['folder'], v['file_name'])
        img = cv2.imread(full_name, 0)
        assert img is not None
        _, des = sift.detectAndCompute(img, None)
        dess[k] = des
    return dess


def most_like_feature(sift, matcher, des, features):
    def get_score(f1, f2):
        matches = matcher.knnMatch(f1, f2, k=2)
        good_matches = [x for x in matches if x[0].distance < 0.7 * x[1].distance]
        good_matches = [good_matches[i][0] for i in range(len(good_matches))]
        if len(good_matches) < 5:
            return 0
        score = len(good_matches) / len(matches)
#         if score < 0.8:
#             return 0
        return score

    sims = []
    for k, v in features.items():
        score = get_score(v, des)
        sims.append((k, score))
    sims = np.array(sims)
    # print('sims shape:', sims.shape)
    idx_most_like = np.argmax(sims[:, 1])
    return sims[idx_most_like, 0]


def test_feature_match(file_dir, images_file=None, annos_file=None):
    import json
    cats_file = os.path.join(cfg.data_dir, 'cats.json')
    print(cats_file)
    with open(cats_file) as f:
        cats = json.load(f)

    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create(extended=True)

    dess_sift = cats_feature(sift, cats)
    dess_surf = cats_feature(surf, cats)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    font = cv2.FONT_HERSHEY_SIMPLEX

    anno_dict = {}
    images_dict = {}
    next_image_id = 1

    for root, _, files in os.walk(file_dir):
        rel_folder = os.path.relpath(root, os.path.dirname(file_dir))
        for file_name in files:
            full_file_name = os.path.join(root, file_name)
            img = cv2.imread(full_file_name)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            assert img is not None

            item = {'folder': rel_folder,
                    'file_name': file_name,
                    'width': img_gray.shape[1],
                    'height': img_gray.shape[0]}
            images_dict[next_image_id] = item

            try:
                m = {}
                x0, y0, w, h, gap, ml, mr = 285, 583, 56, 65, 16, 2, 1
                for i in range(14):
                    x = x0 + w * i + ml
                    x += gap if i == 13 else 0
                    s = img_gray[y0:y0 + h, x:x + w - ml - mr]
                    s_kp, s_des = sift.detectAndCompute(s, None)
#                     print(len(s_kp), len(s_des))
                    if len(s_des) > 10:
                        cat_id = most_like_feature(sift, flann, s_des, dess_sift)
                    else:
                        _, s_des = surf.detectAndCompute(s, None)
                        cat_id = most_like_feature(surf, flann, s_des, dess_surf)
#                         print('using surf:', len(s_des))
    #                 print('most like:', cats[cat_id]['name'])
                    if not cat_id in m:
                        m[cat_id] = {'name': cat_id, 'bbox': []}
                    m[cat_id]['bbox'].append({'x':x, 'y':y0, 'w':w - ml - mr, 'h':h})

#                     cv2.rectangle(img, (x, y0), (x + w-ml-mr, y0+h), (255, 0, 0), 2)
#                     cv2.putText(img, cats[cat_id]['name'], (x, y0), font, 1, (255, 255, 0), 2)
                anno_dict[next_image_id] = m
            except Exception:
                print('sth happened:', file_name)
#             print()
#             cv2.imshow('pic browser', img)
#             k = cv2.waitKey(0)
#             if k == ord('q'):
#                 return
            next_image_id += 1

    if images_dict and images_file:
        save_to_json(images_dict, images_file)
    if anno_dict and annos_file:
        save_to_json(anno_dict, annos_file)


def verify_annotation(data_dir):
    cats = load_json(os.path.join(data_dir, 'cats.json'))
    img_j = load_json(os.path.join(data_dir, 'images.json'))
    anno_j = load_json(os.path.join(data_dir, 'annotations.json'))

    font = cv2.FONT_HERSHEY_SIMPLEX
    for k, v in img_j.items():
        full_name = os.path.join(data_dir, v['folder'], v['file_name'])
        img = cv2.imread(full_name)
        for cat_id in cats.keys():
            anno = anno_j[k]
            assert anno is not None
            if not cat_id in anno:
                continue
            bbs = anno[cat_id]['bbox']
            for b in bbs:
                x, y, w, h = b['x'], b['y'], b['w'], b['h']
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, cats[cat_id]['name'], (x, y), font, 1, (255, 255, 0), 2)

        cv2.imshow('check annotations', img)
        k = cv2.waitKey(0)
        if k == ord('q'):
            return



if __name__ == '__main__':
    cfg.data_dir = '/home/splendor/win/mira/dataset/'
    # big_dir = os.path.join(cfg.data_dir, 'bigpics')
    big_dir = '/home/splendor/jumbo/annlab/mj/bigpics_'
    template = cv2.imread(os.path.join(cfg.data_dir, 'atomitems/b4.png'), 0)
#     browse(big_dir, template)
#     browse2(big_dir, template)
    # test_hist(os.path.join(cfg.data_dir, 'bigpics'))
#     test_feature_match(big_dir, cfg.data_dir + 'images.json', cfg.data_dir + 'annotations.json')
#     verify_annotation(cfg.data_dir)
    # hands = get_pieces('/home/splendor/win/mira/dataset/bigpics/sikuliximage-1493200895127.png')
    # same = np.allclose(hands[4], hands[5])
    # print('piece %d and %d are %ssame' % (4, 5, '' if same else 'not '))
    # get_pieces_batch(big_dir, 'result')
