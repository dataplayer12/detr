import numpy as np


def iou(a, b):
    # a, bは矩形を表すリストで、a=[xmin, ymin, xmax, ymax]
    ax_mn, ay_mn, ax_mx, ay_mx = a[0], a[1], a[2], a[3]
    bx_mn, by_mn, bx_mx, by_mx = b[0], b[1], b[2], b[3]

    a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
    b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)

    abx_mn = max(ax_mn, bx_mn)
    aby_mn = max(ay_mn, by_mn)
    abx_mx = min(ax_mx, bx_mx)
    aby_mx = min(ay_mx, by_mx)
    w = max(0, abx_mx - abx_mn + 1)
    h = max(0, aby_mx - aby_mn + 1)
    intersect = w*h

    iou = intersect / (a_area + b_area - intersect)
    return iou

def nms(bboxes, scores, classes, iou_threshold=0.5):
    if type(bboxes) is not list:
        bboxes = bboxes.tolist()
    if type(scores) is not list:
        scores = scores.tolist()
    if type(classes) is not list:
        classes = classes.tolist()       

    new_bboxes = [] # NMS適用後の矩形リスト
    new_scores = [] # NMS適用後の信頼度(スコア値)リスト
    new_classes = [] # NMS適用後のクラスのリスト

    while len(bboxes) > 0:
        # スコア最大の矩形のインデックスを取得
        argmax = scores.index(max(scores))

        # スコア最大の矩形、スコア値、クラスをそれぞれのリストから消去
        bbox = bboxes.pop(argmax)
        score = scores.pop(argmax)
        clss = classes.pop(argmax)        

        # スコア最大の矩形と、対応するスコア値、クラスをNMS適用後のリストに格納
        new_bboxes.append(bbox)
        new_scores.append(score)
        new_classes.append(clss)

        pop_i = []
        for i, bbox_tmp in enumerate(bboxes):
            # スコア最大の矩形bboxとのIoUがiou_threshold以上のインデックスを取得
            if iou(bbox, bbox_tmp) >= iou_threshold:
                pop_i.append(i)

        # 取得したインデックス(pop_i)の矩形、スコア値、クラスをそれぞれのリストから消去
        for i in pop_i[::-1]:
            bboxes.pop(i)
            scores.pop(i)
            classes.pop(i)

    return np.array(new_bboxes), np.array(new_scores), np.array(new_classes)