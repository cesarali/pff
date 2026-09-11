try:  # pragma: no cover - exercised indirectly via configuration loading
    import yaml  # type: ignore
    from yaml import SafeLoader  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    from pff.config_classes import yaml_fallback as yaml
    SafeLoader = yaml.SafeLoader

class TupleSafeLoader(SafeLoader):
    def construct_python_tuple(self, node):
        # Convert the YAML sequence (e.g., [0.01, 0.1]) into a tuple
        return tuple(self.construct_sequence(node))

# Register the constructor for the fully qualified tag
TupleSafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', TupleSafeLoader.construct_python_tuple)