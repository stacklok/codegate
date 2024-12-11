from urllib.parse import quote


def generate_vector_string(package) -> str:
    vector_str = f"{package['name']}"
    package_url = ""
    type_map = {
        "pypi": "Python package available on PyPI ecosystem",
        "npm": "JavaScript package available on NPM ecosystem",
        "go": "Go package ecosystem",
        "crates": "Rust package available on Crates ecosystem",
        "java": "Java package available on Maven ecosystem",
    }
    status_messages = {
        "archived": "However, this package is found to be archived and no longer maintained.",
        "deprecated": "However, this package is found to be deprecated and no longer "
        "recommended for use.",
        "malicious": "However, this package is found to be malicious and must not be used.",
    }
    vector_str += f" is a {type_map.get(package['type'], 'package of unknown type')}. "
    package_name = quote(package["name"], safe="")
    package_url = f"https://www.insight.stacklok.com/report/{package['type']}/{package_name}"

    # Add extra status
    status_suffix = status_messages.get(package["status"], "")
    if status_suffix:
        vector_str += f" {status_suffix} For additional information refer to {package_url}"

    # add description
    vector_str += f" - Package offers this functionality: {package['description']}"
    return vector_str
