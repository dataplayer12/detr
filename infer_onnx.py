import onnx
import onnxruntime as ort
import cv2
import numpy as np
import pdb
import glob
import os

sess = ort.InferenceSession('detrmodel.onnx', None)

prsize = [768, 480]
det_threshold=0.3
categoy_dict ={1:"a",2:"b"}

def draw_bounding_boxes(cv_img, probs, boxes, cat_ids, categoy_dict):
    cv_img_out = cv_img.copy()
    for prob, (xmin, ymin, xmax, ymax), cat_id, in zip(probs, boxes, cat_ids):
        cv2.rectangle(cv_img_out, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
        text = f"{categoy_dict[cat_id]}: {prob:0.2f}"
        cv2.putText(cv_img_out,
            text,
            org=(xmin, ymin),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_4)
    return cv_img_out


def scale_bboxes(boxesarr):
    xc,yc,w,h=boxesarr[:,0], boxesarr[:,1], boxesarr[:,2], boxesarr[:,3]

    x1 = (xc - 0.5*w)*prsize[0]
    y1 = (yc - 0.5*h)*prsize[1]
    x2 = (xc + 0.5*w)*prsize[0]
    y2 = (yc + 0.5*h)*prsize[1]

    blist=[(int(_x1), int(_y1), int(_x2), int(_y2)) for (_x1, _y1, _x2, _y2) in zip(x1,y1,x2,y2)]
    return blist


# get the name of the first input of the model




input_image_dir = "test_imgs"
img_paths = sorted(glob.glob(os.path.join(input_image_dir, "*.png")))
out_dir = "infer"
os.makedirs(out_dir, exist_ok=True)
for img_path in img_paths:
    input_name = sess.get_inputs()[0].name

    outputs=['logits', 'boxes']
    img=cv2.imread(img_path, cv2.IMREAD_COLOR)[:,:,::-1] #BGR to RGB

    img=cv2.resize(img, (768, 480))

    img_data = img.astype('float32')/255.0
    img_data=img_data.transpose(2,0,1)

    #normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')

    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:] - mean_vec[i]) / stddev_vec[i]

    #add batch channel
    norm_img_data = norm_img_data[None].astype('float32')
    # pdb.set_trace()

    outputs=sess.run(output_names=outputs, input_feed={input_name: norm_img_data})


    _logits= outputs[0][0]
    _boxes= outputs[1][0]
    explogits=np.exp(_logits)
    probs=explogits/(np.sum(explogits, axis=1)[:,None])
    probs=probs[:,:-1]
    tokeep=probs.max(axis=1) >= det_threshold
    probs=probs[tokeep,:]
    cats=probs.argmax(axis=1)
    probs=probs.max(axis=1)
    boxes=scale_bboxes(_boxes[tokeep,:])

    img_with_bb = draw_bounding_boxes(img, probs, boxes, cats, categoy_dict)
    cv2.imwrite(os.path.join(out_dir, os.path.basename(img_path)), img_with_bb)