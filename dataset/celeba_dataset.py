import os.path
from dataset.pix2pix_dataset import Pix2pixDataset
from dataset.image_folder import make_dataset

class CelebaDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=35)
        parser.set_defaults(aspect_ratio=1)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else 'train'

        label_dir = os.path.join(root, 'CelebAMaskHQ-mask-3000', phase)
        label_paths_all = make_dataset(label_dir, recursive=True)
        label_paths = [p for p in label_paths_all]

        color_dir = os.path.join(root, 'CelebAMaskHQ-mask-3000', phase)
        color_paths_all = make_dataset(color_dir, recursive=True)
        color_paths = [p for p in color_paths_all]

        image_dir = os.path.join(root, 'CelebA-HQ-img-3000', phase)
        image_paths = make_dataset(image_dir, recursive=True)

        '''realseg_dir = os.path.join(root, 'seg_pre', phase)
        realseg_paths_all = make_dataset(image_dir, recursive=True)
        realseg_paths = [p for p in color_paths_all if p.endswith('_color.png')]'''

        if not opt.no_instance:
            instance_paths = [p for p in label_paths_all if p.endswith('_instanceIds.png')]
        else:
            instance_paths = []

        # load mask
        mask_dir = os.path.join(root, 'irregular_mask')
        mask_paths_all = make_dataset(mask_dir, recursive=True)
        mask_paths = [p for p in mask_paths_all if p.endswith('.png')]

        return label_paths, image_paths, instance_paths, mask_paths, color_paths 

    def paths_match(self, path1, path2):
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)
        # compare the first 3 components, [city]_[id1]_[id2]
        return '_'.join(name1.split('_')[:3]) == \
            '_'.join(name2.split('_')[:3])
