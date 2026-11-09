"""Download the model, welfare/emotion vectors and Jacobian lens at the revisions the experiment used.
google/gemma-3-27b-it is gated: accept the licence on Hugging Face and log in first."""
import os
from huggingface_hub import snapshot_download

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_REV = "005ad3404e59d6023443cb575daa05336842228a"

snapshot_download("davidafrica/functional-wellbeing", revision="0b005c3da6692912f6bb5a914a5a9d15c4884a91",
                  local_dir=os.path.join(ROOT, "assets", "functional-wellbeing"),
                  allow_patterns=["concept_vectors/cross_model/gemma-3-27b_step325/*/*",
                                  "concept_vectors/emotions_gemma-3-27b/emotion*"])
snapshot_download("neuronpedia/jacobian-lens", revision="d2527c3cdbaa60dd4b65cbca1604921f044e989a",
                  local_dir=os.path.join(ROOT, "assets", "jlens"),
                  allow_patterns=["gemma-3-27b-it/jlens/Salesforce-wikitext/gemma-3-27b-it_jacobian_lens.pt"])
snapshot_download("google/gemma-3-27b-it", revision=MODEL_REV)
print("assets ready")
