class Config:
    """
    A class used to represent configuration
    """
    def __init__(self, config: dict, name: str = "config", level: int = 0):
        self._config = {}
        self._name = name
        self._level = level
        self._init_config(config)

    def _init_config(self, config):
        for key, value in config.items():
            if isinstance(value, dict):
                self._config[key] = Config(
                    value, name=key, level=self._level + 1
                )
            else:
                self._config[key] = value

    def __getattr__(self, attr):
        return self._config.get(attr)

    def __setitem__(self, key, value):
        self._config[key] = value

    def __repr__(self):
        blank_space = " " * 4
        representation = self._level * blank_space + self._name + "={\n"
        for key, value in self._config.items():
            if isinstance(value, Config):
                representation += value.__repr__() + ",\n"
            else:
                representation += (self._level + 1) * blank_space \
                                  + f"{key}={value},\n"
        representation += self._level * blank_space + "}"
        return representation

    def add(self, key, value):
        if isinstance(value, dict):
            self._config[key] = Config(value, name=key, level=self._level + 1)
        else:
            self._config[key] = value

    def get(self, key, default_value):
        return self._config.get(key, default_value)

    def to_dict(self):
        config_dict = {}
        for key, value in self._config.items():
            if isinstance(value, Config):
                config_dict[key] = value.to_dict()
            else:
                config_dict[key] = value
        return config_dict

    def keys(self):
        return self._config.keys()
