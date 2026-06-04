from promptlens.adapters import EchoAdapter
from promptlens.mutators import LLMRewriteMutator
from promptlens.segmenters import SentenceSegmenter


def test_llm_rewrite_mutator_marks_each_feature() -> None:
    features = SentenceSegmenter().segment("Alpha sentence. Beta sentence.")
    mutator = LLMRewriteMutator(EchoAdapter(model="rewrite"), rewrites_per_feature=1)

    mutations = mutator.mutate("Alpha sentence. Beta sentence.", features)

    assert len(mutations) == 2
    assert mutations[0].feature == features[0]
    assert "<mutate>Alpha sentence.</mutate>" in mutations[0].prompt
    assert mutations[0].metadata["rewrite_model"] == "rewrite"
