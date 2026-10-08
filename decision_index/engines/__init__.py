from decision_index.engines.base import Engine, NativeAbstention, RandomEngine, Unsupported, validate

REGISTRY = {
    "http": "decision_index.engines.http:HttpSystemOne",
    "transformers": "decision_index.engines.transformers_engine:TransformersEngine",
    "random": "decision_index.engines.base:RandomEngine",
    "vllm-vl": "decision_index.vision.engines:VllmVisionChat",
    "decide": "decision_index.vision.engines:HttpDecide",
    "openai-vlm": "decision_index.vision.engines:OpenAIVision",
}


def load_engine(name, **options):
    import importlib

    target = REGISTRY.get(name, name)
    if ":" not in target:
        raise KeyError(f"unknown engine {name!r}; use one of {sorted(REGISTRY)} or a dotted module:Class path")
    module, cls = target.split(":")
    return getattr(importlib.import_module(module), cls)(**options)


__all__ = ["Engine", "RandomEngine", "Unsupported", "NativeAbstention", "validate", "load_engine", "REGISTRY"]
