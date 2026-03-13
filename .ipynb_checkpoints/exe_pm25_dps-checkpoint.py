import argparse
import torch
import datetime
import json
import yaml
import os
import random
import numpy as np


from dataset_pm25 import get_dataloader
from main_model_dps_Copy3 import CSDI_PM25
from utils_dps_Copy3 import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument(
    "--targetstrategy", type=str, default="mix", choices=["mix", "random", "historical"]
)
parser.add_argument(
    "--validationindex", type=int, default=0, help="index of month used for validation (value:[0-7])"
)
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--dps_scale", type=float, default=1.0)
parser.add_argument("--left_window", type=int, default=1)
parser.add_argument("--right_window", type=int, default=1)
parser.add_argument("--f_low", type=float, default=0.07)
parser.add_argument("--f_high", type=float, default=0.20)



args = parser.parse_args()
print(args)



path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["target_strategy"] = args.targetstrategy
config["dps"] = {
    "dps_scale": args.dps_scale,
    "left_window": args.left_window,
    "right_window": args.right_window,
    "f_low": args.f_low,
    "f_high": args.f_high,
}

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
foldername = (
    f"./save/run{args.validationindex}/")

print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    config["train"]["batch_size"],
    device=args.device,
    validindex=args.validationindex,
)

model = CSDI_PM25(config, args.device).to(args.device)
model.dps_scale = args.dps_scale
model.left_window = args.left_window
model.right_window = args.right_window
model.f_low = args.f_low
model.f_high = args.f_high




if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("../save/pm_validation/" + args.modelfolder + "/model.pth"))

evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
)
