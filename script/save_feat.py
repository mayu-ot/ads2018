import chainer
from chainer import cuda
import chainercv
import numpy as np
import tables
from chainercv import utils
from chainercv.visualizations import vis_bbox
from chainercv.datasets import voc_bbox_label_names
from chainercv.links.model.faster_rcnn import FasterRCNNVGG16
import chainer.functions as F
import os
import progressbar

class FasterRCNNVGG16Extractor(FasterRCNNVGG16):
    def __call__(self, x, scale=1.):
        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores, fc7 = self.extract_head(
            h, rois, roi_indices)
        
        return roi_cls_locs, roi_scores, fc7, rois, roi_indices
    
    def extract_head(self, x, rois, roi_indices):
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)
        pool = _roi_pooling_2d_yx(
            x, indices_and_rois, self.head.roi_size, self.head.roi_size,
            self.head.spatial_scale)

        fc6 = F.relu(self.head.fc6(pool))
        fc7 = F.relu(self.head.fc7(fc6))
        roi_cls_locs = self.head.cls_loc(fc7)
        roi_scores = self.head.score(fc7)
        return roi_cls_locs, roi_scores, fc7

def _roi_pooling_2d_yx(x, indices_and_rois, outh, outw, spatial_scale):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    pool = F.roi_pooling_2d(
        x, xy_indices_and_rois, outh, outw, spatial_scale)
    return pool

if __name__ == '__main__':
    VA_DATASET_ROOT = os.getenv('VA_DATASET_ROOT')
    
    device = 0
    
    model = FasterRCNNVGG16Extractor(n_fg_class=len(voc_bbox_label_names),
                            min_size=150,
                            pretrained_model='voc0712',
                            proposal_creator_params={'n_test_pre_nms':1000,
                                                     'n_test_post_nms':10})

    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    for split in ['train', 'test']:
        h5file = tables.open_file('data/frcnn_feat/%s.h5' % split, 'w')

        images = []
        for root, dirs, files in os.walk(VA_DATASET_ROOT+'%s_images/' % split):
            dir_n = root.split('/')[-1]

            if '/'+dir_n not in h5file:
                h5file.create_group('/', dir_n)

            print(dir_n)

            for name in files:
                im = utils.read_image(os.path.join(root, name), color=True)

                with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                    x = model.prepare(im)
                    x = cuda.to_gpu(x[None, :])
                    roi_cls_locs, roi_scores, fc7, rois, roi_indices = model(x)

                fc7.to_cpu()
                rois= cuda.to_cpu(rois)

                h5file.create_group('/'+dir_n, name.split('.')[0])
                h5file.create_array('/%s/%s' % (dir_n, name.split('.')[0]), 'feat', obj=fc7.data)
                h5file.create_array('/%s/%s' % (dir_n, name.split('.')[0]), 'roi', obj=rois)

        h5file.close()