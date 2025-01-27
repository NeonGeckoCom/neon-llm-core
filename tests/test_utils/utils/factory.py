import uuid

from neon_data_models.models import LLMPersona


class PersonaFactory:

    @staticmethod
    def _random_uuid_hex() -> str:
        return uuid.uuid4().hex

    @classmethod
    def create_mock_llm_persona(cls, enabled: bool = False):
        return LLMPersona(
            name=f"mock_persona_{cls._random_uuid_hex()}",
            user_id=None,
            description=f"Mock Persona {cls._random_uuid_hex()}",
            system_prompt=f"Mock Prompt: {cls._random_uuid_hex()}",
            enabled=enabled,
        )
