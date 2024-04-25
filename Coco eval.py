from pycocotools.coco import COCO 
from pycocotools.cocoeval import COCOeval
 
cocoGt = COCO('/Users/mohitkumar/Downloads/Forest Fire.v4i.coco 2/valid/_annotations.coco.json') 
cocoDt = cocoGt.loadRes('/Users/mohitkumar/Downloads/Forest Fire.v4i.coco 2/valid/filtered_data_all_images_improvement_GPU.json')  

coco_eval = COCOeval(cocoGt, cocoDt, 'bbox')


imgIds = sorted(cocoGt.getImgIds())
coco_eval.params.imgIds = imgIds 

coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
