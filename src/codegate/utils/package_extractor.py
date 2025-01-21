import structlog
import tree_sitter_go as tsgo
import tree_sitter_java as tsjava
import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython
import tree_sitter_rust as tsrust
from tree_sitter import Language, Parser

logger = structlog.get_logger("codegate")


class PackageExtractor:
    __languages = {
        "javascript": Language(tsjavascript.language()),
        "go": Language(tsgo.language()),
        "python": Language(tspython.language()),
        "java": Language(tsjava.language()),
        "rust": Language(tsrust.language()),
    }
    __parsers = {
        "javascript": Parser(__languages["javascript"]),
        "go": Parser(__languages["go"]),
        "python": Parser(__languages["python"]),
        "java": Parser(__languages["java"]),
        "rust": Parser(__languages["rust"]),
    }
    __queries = {
        "javascript": """
                    (import_statement
                        source: (string) @import_name)
                    (call_expression
                        function: (identifier) @require
                        arguments: (arguments (string) @import_name)
                        (#eq? @require "require")
                    )
                """,
        "go": """
                    (import_declaration
                        (import_spec
                            (interpreted_string_literal) @import_name
                        )
                    )
                    (import_declaration
                        (import_spec_list
                            (import_spec
                                (interpreted_string_literal) @import_name
                            )
                        )
                    )
                """,
        "python": """
                    (import_statement
                        name: (dotted_name) @import_name)
                    (import_from_statement
                        module_name: (dotted_name) @import_name)
                    (import_statement
                        (aliased_import (dotted_name) @import_name (identifier)))
                """,
        "java": """
                    (import_declaration
                        (scoped_identifier) @import_name)
                """,
        "rust": """
                    (use_declaration
                        (scoped_identifier) @import_name)
                    (use_declaration
                        (identifier) @import_name)
                    (use_declaration
                        (use_wildcard) @import_name)
                    (use_declaration
                        (use_as_clause (scoped_identifier) @import_name))
                """,
    }

    @staticmethod
    def extract_packages(code: str, language_name: str) -> list[str]:
        if (code is None) or (language_name is None):
            return []

        language_name = language_name.lower()

        if language_name not in PackageExtractor.__languages.keys():
            return []

        language = PackageExtractor.__languages[language_name]
        parser = PackageExtractor.__parsers[language_name]

        # Create tree
        tree = parser.parse(bytes(code, "utf8"))

        # Create query for imports
        query = language.query(PackageExtractor.__queries[language_name])

        # Execute query
        all_captures = query.captures(tree.root_node)

        # Collect imports
        imports = set()
        for capture_name, captures in all_captures.items():
            if capture_name != "import_name":
                continue
            for capture in captures:
                import_lib = code[capture.start_byte : capture.end_byte]

                # Remove quotes from the import string
                import_lib = import_lib.strip("'\"")

                # Get the root library name
                if language_name == "python":
                    import_lib = import_lib.split(".")[0]

                if language_name == "rust":
                    import_lib = import_lib.split("::")[0]

                imports.add(import_lib)

        return list(imports)
