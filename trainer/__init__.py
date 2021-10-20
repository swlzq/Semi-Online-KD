from trainer.sokd import SemiOnlineKnowledgeDistillation
from trainer.vanilla import Vanilla



def build_trainer(**kwargs):
    maps = dict(
        sokd=SemiOnlineKnowledgeDistillation,
        vanilla=Vanilla,
    )
    return maps[kwargs['distillation_type']](kwargs)

