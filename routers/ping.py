from typing import Dict, Optional, Union, List

from fastapi import APIRouter
from pydantic import BaseModel

from services.claster_func import get_clusters

router = APIRouter()

@router.get("/ping", summary="Ping-Pong API", tags=["Health Check"])
def ping():
    """
    Простая проверка доступности сервиса.
    Возвращает 'pong'.
    """
    return {"message": "pong"}

# Входная модель
class TextInput(BaseModel):
    text: str

# Все возможные ключи
id_to_label = {
    0: 'DOC',
    1: 'MDT',
    2: 'NAME',
    3: 'O',
    4: 'ORG',
    5: 'POS',
    6: 'TEL',
    7: 'VOL',
}

@router.post("/ner", summary="Named Entity Recognition", tags=["NER"])
def extract_entities(input_data: TextInput) -> Dict[str, Optional[Union[List[str], None]]]:
    # Получаем результат из модели
    result = get_clusters(input_data.text)

    # Добавляем отсутствующие ключи с None
    full_result = {
        label: result.get(label) if label in result else None
        for label in id_to_label.values()
    }

    return full_result