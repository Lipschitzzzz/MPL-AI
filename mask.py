import torch

model_path = 'checkpoints/200 epoch 1 month.pth'
# model_path = 'checkpoints/10.31--17.38 model.pth'

try:
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
except Exception as e:
    print(f"error: {e}")
    exit()

possible_keys = ['config', 'hparams', 'args', 'hyper_parameters', 'settings']

for key in possible_keys:
    if key in checkpoint:
        config = checkpoint[key]
        if isinstance(config, dict):
            if 'criterion' in config:
                print(f"loss function (criterion): {config['criterion']}")
            elif 'loss' in config:
                print(f"loss function (loss): {config['loss']}")
            elif 'loss_fn' in config:
                print(f"loss function (loss_fn): {config['loss_fn']}")
            else:
                for k, v in config.items():
                    if 'loss' in k.lower() or 'crit' in k.lower():
                        print(f"{k}: {v}")
        elif hasattr(config, '__dict__'):
            for k, v in config.__dict__.items():
                if 'loss' in k.lower() or 'crit' in k.lower():
                    print(f"{k}: {v}")

print("type:", type(checkpoint))

state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict')
if state_dict:
    print(state_dict.keys())
    print("architecture:")
    for name in state_dict.keys():
        print(name)

if isinstance(checkpoint, dict):
    print("This is a dictionary")
    
    common_keys = ['state_dict', 'model', 'optimizer', 'epoch', 'loss', 'args', 'hyper_parameters']
    
    for key in common_keys:
        if key in checkpoint:
            print(f" key '{key}':")
            if key == 'state_dict' or key == 'model':
                state_dict = checkpoint[key]
                print(f" include {len(state_dict)} parameters")
                for name, param in state_dict.items():
                    if hasattr(param, 'shape'):
                        print(f" {name}: shape={list(param.shape)}, dtype={param.dtype}")
                    else:
                        print(f" {name}: type={type(param)}, value={param}")
            elif key == 'epoch' or key == 'loss':
                print(f" {key}: {checkpoint[key]}")
            else:
                print(f" {key}: available (type: {type(checkpoint[key])})")
                if isinstance(checkpoint[key], dict):
                    print(f" content: {checkpoint[key]}")
                elif hasattr(checkpoint[key], '__dict__'):
                    print(f" attribute: {dir(checkpoint[key])}")
    else:
        print("all keys")
        for key in checkpoint.keys():
            value = checkpoint[key]
            print(f" key: {key}")
            if hasattr(value, 'shape'):
                print(f" shape: {list(value.shape)}, data type: {value.dtype}")
            elif isinstance(value, torch.Tensor):
                print(f" is a tensor, data type: {value.dtype}")
            elif isinstance(value, (int, float, str, bool)):
                print(f" value: {value}")
            else:
                print(f" type: {type(value)}")

elif isinstance(checkpoint, torch.nn.Module):
    print("This is a PyTorch model")
    print("Model Architecture")
    print(checkpoint)
    
    print("model state_dict:")
    state_dict = checkpoint.state_dict()
    for name, param in state_dict.items():
        print(f" {name}: shape={list(param.shape)}, dtype={param.dtype}")

elif isinstance(checkpoint, torch.Tensor):
    print("This is a single tensor")
    print(f" shape: {list(checkpoint.shape)}, data type: {checkpoint.dtype}")

else:
    print(f" content type: {type(checkpoint)} ---")
    print("Content:", checkpoint)