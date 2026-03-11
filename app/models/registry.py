class ModelRegistry:
    _models = {}

    @classmethod
    def register(cls, name: str, model):
        cls._models[name] = model

    @classmethod
    def get(cls, name: str):
        return cls._models.get(name)
