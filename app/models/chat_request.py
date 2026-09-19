from pydantic import BaseModel, field_validator

class ChatRequest(BaseModel):
    query: str

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Query must be at least 2 characters long.")
        if len(v) > 500:
            raise ValueError("Query must not exceed 500 characters.")
        return v