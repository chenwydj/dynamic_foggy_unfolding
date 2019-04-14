import time
from EnlightenGAN.options.train_options import TrainOptions
from EnlightenGAN.data.data_loader import CreateDataLoader
from EnlightenGAN.models.models import create_model
from EnlightenGAN.util.visualizer import Visualizer
import torch

def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.load(stream)

opt = TrainOptions().parse()
config = get_config(opt.config)
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = 0


##### BDD Seg. #################################################
import bdd.encoding.utils as utils_seg
from bdd.encoding.nn import SegmentationLosses, BatchNorm2d
from bdd.encoding.nn import SegmentationMultiLosses
from utils.focal_loss import FocalLoss
from bdd.encoding.parallel import DataParallelModel, DataParallelCriterion
from bdd.encoding.models import get_segmentation_model
from bdd.experiments.segmentation.option import Options

opt_seg = Options()#.parse()
opt_seg.dataset = "cityscapes"
opt_seg.model = "fcn"
opt_seg.backbone = "resnet50"
opt_seg.dilated = True
opt_seg.ft = True
opt_seg.ft_resume = "/home/chenwy/DynamicDeHaze/cityscapes/fcn_model/res50_di_mg_500px/model_best.pth.tar"
opt_seg.eval = True

seg = get_segmentation_model(opt_seg.model, dataset=opt_seg.dataset, backbone=opt_seg.backbone, aux=False, se_loss=False,
                               dilated=opt_seg.dilated,
                               # norm_layer=BatchNorm2d, # for multi-gpu
                               base_size=720, crop_size=500, multi_grid=True, multi_dilation=[2,4,8])
# seg = DataParallelModel(seg).cuda()
seg = torch.nn.DataParallel(seg).cuda()
seg.eval()
if opt_seg.ft:
    checkpoint = torch.load(opt_seg.ft_resume)
    seg.module.load_state_dict(checkpoint['state_dict'], strict=False)

# seg_criterion = SegmentationLosses(se_loss=False, aux=False, nclass=19)
# seg_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
seg_criterion = FocalLoss(gamma=5, ignore=-1)
seg_criterion = seg_criterion.cuda()
####################################################


for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data)
        model.optimize_parameters(epoch, i, seg=seg, seg_criterion=seg_criterion)

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            print(opt.name)
            errors = model.get_current_errors(epoch)
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if opt.new_lr:
        if epoch == opt.niter:
            model.update_learning_rate()
        elif epoch == (opt.niter + 20):
            model.update_learning_rate()
        elif epoch == (opt.niter + 70):
            model.update_learning_rate()
        elif epoch == (opt.niter + 90):
            model.update_learning_rate()
            model.update_learning_rate()
            model.update_learning_rate()
            model.update_learning_rate()
    else:
        if epoch > opt.niter:
            model.update_learning_rate()
