import torch
import torchvision.models as models
import pickle as pkl
import sys


"""
Usage:
  # download pretrained dinov2 model:
  wget -P ./ckpt https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth
  # run the conversion
  python utils/convert-resnet-to-dinov2.py
  # Then, use dinov2_vitg14_pretrain_updated.pkl with the following changes in config:
MODEL:
  WEIGHTS: "/path/to/dinov2_vitg14_pretrain_updated.pkl"
INPUT:
  FORMAT: "RGB"
"""
def adjust_keys(state_dict):
    new_state_dict = {}
    for key in state_dict.keys():
        parts = key.split('.')

        if len(parts) > 2:
            if parts[2] == 'ls1':
                new_key = f'blocks.{parts[1]}.' + 'gamma1'
                new_state_dict[new_key] = state_dict[key]
            elif parts[2] == 'ls2':
                new_key = f'blocks.{parts[1]}.' + 'gamma2'
                new_state_dict[new_key] = state_dict[key]
            elif parts[0] == 'pos_embed':
                new_key = f'backbone' + '.'+'pos_embed'
                new_state_dict[new_key] = state_dict[key]

            else:
                new_state_dict[key] = state_dict[key]
        elif parts[0] == 'pos_embed':
            new_key = f'backbone' + '.'+'pos_embed'
            new_state_dict[new_key] = state_dict[key]
        elif parts[0] == 'norm':
            new_key = f'backbone' + '.'+ key
            new_state_dict[new_key] = state_dict[key]
            print(state_dict[key])
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict
if __name__ == "__main__":

    input = sys.argv[1]
    obj = torch.load(input, map_location="cpu")
    
    obj = adjust_keys(obj) 


    dinov2_state_dict = {"model": obj, "__author__": "third_party", "matching_heuristics": True}

    resnet50 = models.resnet50(pretrained=True)
    resnet_state_dict = resnet50.state_dict()

    new_params = {"backbone.edge."+k: v for k, v in resnet_state_dict.items() if k!="fc.weight" and k!="fc.bias"}
    dinov2_state_dict['model'].update(new_params)

    new_dinov2_checkpoint_path = sys.argv[2]
    with open(new_dinov2_checkpoint_path, 'wb') as f:
        pkl.dump(dinov2_state_dict, f)

    print(f"Updated DINOv2 parameters saved to {new_dinov2_checkpoint_path}")
