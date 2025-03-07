import unicodedata
import uuid
from typing import List, Optional

import numpy as np
import regex as re
import structlog

from codegate.config import Config
from codegate.db import models as db_models
from codegate.db.connection import DbReader, DbRecorder
from codegate.inference.inference_engine import LlamaCppInferenceEngine

logger = structlog.get_logger("codegate")


REMOVE_URLS = re.compile(r"https?://\S+|www\.\S+")
REMOVE_EMAILS = re.compile(r"\S+@\S+")
REMOVE_CODE_BLOCKS = re.compile(r"```[\s\S]*?```")
REMOVE_INLINE_CODE = re.compile(r"`[^`]*`")
REMOVE_HTML_TAGS = re.compile(r"<[^>]+>")
REMOVE_PUNCTUATION = re.compile(r"[^\w\s\']")
NORMALIZE_WHITESPACE = re.compile(r"\s+")
NORMALIZE_DECIMAL_NUMBERS = re.compile(r"\b\d+\.\d+\b")
NORMALIZE_INTEGER_NUMBERS = re.compile(r"\b\d+\b")


class PersonaDoesNotExistError(Exception):
    pass


class PersonaSimilarDescriptionError(Exception):
    pass


class PersonaManager:

    def __init__(self):
        Config.load()
        conf = Config.get_config()
        self._inference_engine = LlamaCppInferenceEngine()
        self._embeddings_model = f"{conf.model_base_path}/{conf.embedding_model}"
        self._n_gpu = conf.chat_model_n_gpu_layers
        self._persona_threshold = conf.persona_threshold
        self._persona_diff_desc_threshold = conf.persona_diff_desc_threshold
        self._db_recorder = DbRecorder()
        self._db_reader = DbReader()

    def _clean_text_for_embedding(self, text: str) -> str:
        """
        Clean the text for embedding. This function should be used to preprocess the text
        before embedding.

        Performs the following operations:
        1. Replaces newlines and carriage returns with spaces
        2. Removes extra whitespace
        3. Converts to lowercase
        4. Removes URLs and email addresses
        5. Removes code block markers and other markdown syntax
        6. Normalizes Unicode characters
        7. Handles special characters and punctuation
        8. Normalizes numbers
        """
        if not text:
            return ""

        # Replace newlines and carriage returns with spaces
        text = text.replace("\n", " ").replace("\r", " ")

        # Normalize Unicode characters (e.g., convert accented characters to ASCII equivalents)
        text = unicodedata.normalize("NFKD", text)
        text = "".join([c for c in text if not unicodedata.combining(c)])

        # Remove URLs
        text = REMOVE_URLS.sub(" ", text)

        # Remove email addresses
        text = REMOVE_EMAILS.sub(" ", text)

        # Remove code block markers and other markdown/code syntax
        text = REMOVE_CODE_BLOCKS.sub(" ", text)
        text = REMOVE_INLINE_CODE.sub(" ", text)

        # Remove HTML/XML tags
        text = REMOVE_HTML_TAGS.sub(" ", text)

        # Normalize numbers (replace with placeholder)
        text = NORMALIZE_DECIMAL_NUMBERS.sub(" NUM ", text)  # Decimal numbers
        text = NORMALIZE_INTEGER_NUMBERS.sub(" NUM ", text)  # Integer numbers

        # Replace punctuation with spaces (keeping apostrophes for contractions)
        text = REMOVE_PUNCTUATION.sub(" ", text)

        # Normalize whitespace (replace multiple spaces with a single space)
        text = NORMALIZE_WHITESPACE.sub(" ", text)

        # Convert to lowercase and strip
        text = text.strip()

        return text

    async def _embed_text(self, text: str) -> np.ndarray:
        """
        Helper function to embed text using the inference engine.
        """
        cleaned_text = self._clean_text_for_embedding(text)
        # .embed returns a list of embeddings
        embed_list = await self._inference_engine.embed(
            self._embeddings_model, [cleaned_text], n_gpu_layers=self._n_gpu
        )
        # Use only the first entry in the list and make sure we have the appropriate type
        logger.debug("Text embedded in semantic routing", text=cleaned_text[:50])
        return np.array(embed_list[0], dtype=np.float32)

    async def _is_persona_description_diff(
        self, emb_persona_desc: np.ndarray, exclude_id: Optional[str]
    ) -> bool:
        """
        Check if the persona description is different enough from existing personas.
        """
        # The distance calculation is done in the database
        persona_distances = await self._db_reader.get_distance_to_existing_personas(
            emb_persona_desc, exclude_id
        )
        if not persona_distances:
            return True

        for persona_distance in persona_distances:
            logger.info(
                f"Persona description distance to {persona_distance.name}",
                distance=persona_distance.distance,
            )
            # If the distance is less than the threshold, the persona description is too similar
            if persona_distance.distance < self._persona_diff_desc_threshold:
                return False
        return True

    async def _validate_persona_description(
        self, persona_desc: str, exclude_id: str = None
    ) -> np.ndarray:
        """
        Validate the persona description by embedding the text and checking if it is
        different enough from existing personas.
        """
        emb_persona_desc = await self._embed_text(persona_desc)
        if not await self._is_persona_description_diff(emb_persona_desc, exclude_id):
            raise PersonaSimilarDescriptionError(
                "The persona description is too similar to existing personas."
            )
        return emb_persona_desc

    async def add_persona(self, persona_name: str, persona_desc: str) -> None:
        """
        Add a new persona to the database. The persona description is embedded
        and stored in the database.
        """
        emb_persona_desc = await self._validate_persona_description(persona_desc)

        new_persona = db_models.PersonaEmbedding(
            id=str(uuid.uuid4()),
            name=persona_name,
            description=persona_desc,
            description_embedding=emb_persona_desc,
        )
        await self._db_recorder.add_persona(new_persona)
        logger.info(f"Added persona {persona_name} to the database.")

    async def get_persona(self, persona_name: str) -> db_models.Persona:
        """
        Get a persona from the database by name.
        """
        persona = await self._db_reader.get_persona_by_name(persona_name)
        if not persona:
            raise PersonaDoesNotExistError(f"Persona {persona_name} does not exist.")
        return persona

    async def get_all_personas(self) -> List[db_models.Persona]:
        """
        Get all personas from the database.
        """
        return await self._db_reader.get_all_personas()

    async def update_persona(
        self, persona_name: str, new_persona_name: str, new_persona_desc: str
    ) -> None:
        """
        Update an existing persona in the database. The name and description are
        updated in the database, but the ID remains the same.
        """
        # First we check if the persona exists, if not we raise an error
        found_persona = await self._db_reader.get_persona_by_name(persona_name)
        if not found_persona:
            raise PersonaDoesNotExistError(f"Person {persona_name} does not exist.")

        emb_persona_desc = await self._validate_persona_description(
            new_persona_desc, exclude_id=found_persona.id
        )

        # Then we update the attributes in the database
        updated_persona = db_models.PersonaEmbedding(
            id=found_persona.id,
            name=new_persona_name,
            description=new_persona_desc,
            description_embedding=emb_persona_desc,
        )
        await self._db_recorder.update_persona(updated_persona)
        logger.info(f"Updated persona {persona_name} in the database.")

    async def delete_persona(self, persona_name: str) -> None:
        """
        Delete a persona from the database.
        """
        persona = await self._db_reader.get_persona_by_name(persona_name)
        if not persona:
            raise PersonaDoesNotExistError(f"Persona {persona_name} does not exist.")

        await self._db_recorder.delete_persona(persona.id)
        logger.info(f"Deleted persona {persona_name} from the database.")

    async def check_persona_match(self, persona_name: str, query: str) -> bool:
        """
        Check if the query matches the persona description. A vector similarity
        search is performed between the query and the persona description.
        0 means the vectors are identical, 2 means they are orthogonal.
        See
        [sqlite docs](https://alexgarcia.xyz/sqlite-vec/api-reference.html#vec_distance_cosine)
        """
        persona = await self._db_reader.get_persona_by_name(persona_name)
        if not persona:
            raise PersonaDoesNotExistError(f"Persona {persona_name} does not exist.")

        emb_query = await self._embed_text(query)
        persona_distance = await self._db_reader.get_distance_to_persona(persona.id, emb_query)
        logger.info(f"Persona distance to {persona_name}", distance=persona_distance.distance)
        if persona_distance.distance < self._persona_threshold:
            return True
        return False
