from detectron2.config import CfgNode as CN


def add_focus_config(cfg):
    """
    Add config for FOCUS.
    """
    cfg.INPUT.DATASET_MAPPER_NAME = "focus_dataset_mapper"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    # wandb
    cfg.WANDB = CN()
    cfg.WANDB.PROJECT = "FOCUS"
    cfg.WANDB.NAME = None
    cfg.TEST.WANDBENABLE = False
    cfg.MODEL.IS_TRAIN = True
    cfg.MODEL.IS_DEMO = False
    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.CLIP_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.L2_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.COS_WEIGHT = 3.0
    cfg.MODEL.MASK_FORMER.BBOX_WEIGHT = 3.0
    cfg.MODEL.MASK_FORMER.GIOU_WEIGHT = 3.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.MASK_FORMER.TEXT_FEATURE= []

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"
    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

def add_dinov2_config(cfg):
    """
    Add config for DINOv2 Backbone.
    """
    # dinov2 transformer backbone
    cfg.MODEL.DINOV2 = CN()
    cfg.MODEL.DINOV2.NUM_HEADS = 12
    cfg.MODEL.DINOV2.CONV_INPLANE = 64
    cfg.MODEL.DINOV2.N_POINTS = 4
    cfg.MODEL.DINOV2.IN_PATCH_SIZE = 4
    cfg.MODEL.DINOV2.EMBED_DIM = 64
    cfg.MODEL.DINOV2.DEPTHS = [3, 3, 27, 3]
    cfg.MODEL.DINOV2.DEPTH = 40
    cfg.MODEL.DINOV2.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.DINOV2.DEFORM_NUM_HEADS = 6
    cfg.MODEL.DINOV2.FFN_TYPE = "swiglu"

    cfg.MODEL.DINOV2.INIT_VALUES = 0.0
    cfg.MODEL.DINOV2.WITH_CFFN = True
    cfg.MODEL.DINOV2.CFFN_RATIO = 0.25

    cfg.MODEL.DINOV2.DEFORM_RATIO = 1.0
    cfg.MODEL.DINOV2.ADD_VIT_FEATURE = True
    cfg.MODEL.DINOV2.PRETRAINED = None
    cfg.MODEL.DINOV2.USE_EXTRA_EXTRATOR = True
    cfg.MODEL.DINOV2.FREEZE_VIT = False
    cfg.MODEL.DINOV2.USE_CLS = True
    cfg.MODEL.DINOV2.WITH_CP = False

    cfg.MODEL.DINOV2.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.DINOV2.INTERACTION_INDEXES=[[0, 9], [10, 19], [20, 29], [30, 39]]
def add_cliprefiner_config(cfg):
    """
    Add config for CLIP Refiner.
    """

    # FOCAL transformer backbone
    cfg.MODEL.CLIP_REFINER = CN()
    cfg.MODEL.CLIP_REFINER.TEXT_TEMPLATES = "vild"
    # for predefined
    cfg.MODEL.CLIP_REFINER.PREDEFINED_PROMPT_TEMPLATES = ["a photo of a {}."]
    # for learnable prompt
    cfg.MODEL.CLIP_REFINER.PROMPT_CHECKPOINT = ""
    cfg.MODEL.CLIP_REFINER.CLIP_MODEL_NAME = "ViT-B/16"
    cfg.MODEL.CLIP_REFINER.MASK_FILL = "mean"
    cfg.MODEL.CLIP_REFINER.MASK_EXPAND_RATIO = 1.0
    cfg.MODEL.CLIP_REFINER.MASK_THR = 0.4
    cfg.MODEL.CLIP_REFINER.MASK_MATTING = False
    cfg.MODEL.CLIP_REFINER.REGION_RESIZED = True
    cfg.MODEL.CLIP_REFINER.CLIP_ENSEMBLE = True
    cfg.MODEL.CLIP_REFINER.CLIP_ENSEMBLE_WEIGHT = 0.7
    # for mask prompt
    cfg.MODEL.CLIP_REFINER.MASK_PROMPT_DEPTH = 3
    cfg.MODEL.CLIP_REFINER.MASK_PROMPT_FWD = False
    cfg.MODEL.CLIP_REFINER.PROMPTS = []