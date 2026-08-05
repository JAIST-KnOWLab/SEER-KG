from .counterfact import CounterFactDataset
from .zsre import ZsreDataset
try:
    # Multimodal datasets pull vision deps (opencv/cv2, torchvision). Optional —
    # not needed for text editing (ROME/MEND/IKE/RAG).
    from .coco_caption import CaptionDataset
    from .vqa import VQADataset
except ImportError as _e:
    import warnings as _warnings
    _warnings.warn(f"easyeditor: multimodal datasets unavailable ({_e}); text editing still works.")
from .wiki_recent import WikiRecentDataset
from .knowedit import KnowEditDataset
from .sanitization import SanitizationTrainDataset
from .multitask import MultiTaskDataset
from .personality import PersonalityDataset
from .safety import SafetyDataset
from .Cknowedit import CKnowEditDataset
from .MQuAKE import MQuAKEDataset
from .wikibigedit import WikiBigEditDataset
