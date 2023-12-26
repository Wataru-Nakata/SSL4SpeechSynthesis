from ssl4speechsynthesis.hf_model.necobert.configuration_necobert import NecoBertConfig
from ssl4speechsynthesis.hf_model.necobert.modeling_necobert import NecoBertModel,NecoBertTokenizer    
from ssl4speechsynthesis.model.lightning_module import DACBertLightningModule
from ssl4speechsynthesis.model.lightning_module import DACBertLightningModule
import transformers
import torch
import dac

dac_path = "../../descript-audio-codec/runs/baseline_50Hz_librispeech/best/dac/weights.pth"
config = NecoBertConfig(n_heads=1,max_position_embeddings=4096)
model = NecoBertModel(config)
params = torch.load("./ssl4speechsynthesis/uee8vdmp/checkpoints/epoch=11-step=130000.ckpt",map_location="cpu")
pretrained_dict = {k.removeprefix('model.'): v for k, v in params['state_dict'].items() if k in model.state_dict()}
model.model.load_state_dict(pretrained_dict)
model.preprocessor.dac= dac.DAC.load(dac_path)
NecoBertConfig.register_for_auto_class()
NecoBertModel.register_for_auto_class("AutoModel")
model.save_pretrained("./necobert-base-masked")
