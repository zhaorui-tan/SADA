# from diffusers import DiffusionPipeline
# from pipeline import MyScoreSdeVePipeline
# DiffusionPipeline.ScoreSdeVePipeline = MyScoreSdeVePipeline
# import pipeline
# import diffusers_local.src.diffusers as diffusers
from diffusers import DiffusionPipeline
import diffusers

# from transformers import CLIPFeatureExtractor, CLIPModel
#
# import test_SDE_pipline
#
# clip_model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
#
# feature_extractor = CLIPFeatureExtractor.from_pretrained(clip_model_id)
# clip_model = CLIPModel.from_pretrained(clip_model_id)


model_id = "google/ncsnpp-ffhq-256"

# load model and scheduler
sde_ve = DiffusionPipeline.from_pretrained(model_id).to("cuda")
# run pipeline in inference (sample random noise and denoise)
image = sde_ve().images[0]


# save image
image.save("sde_ve_generated_image.png")