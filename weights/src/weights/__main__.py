import torch
import glob
import os

for file in glob.glob("input/*.pth"):
  output_path=f"output/{os.path.basename(file)}"
  print(f"Converting {file} -> {output_path}")
  try:
      weights=torch.load(file)
      w={}
      for param_tensor in weights:
          w[param_tensor]=weights[param_tensor].contiguous()
          # print(param_tensor, "\t", weights[param_tensor].size(), "\t", weights[param_tensor].stride())
      torch.save(w, output_path)
  except:
     print(file, "failed")
