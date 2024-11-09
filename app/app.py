import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI, HTTPException

from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load your trained model and encoders
model = joblib.load("ml/knn_new_model.joblib")
vectorizer = joblib.load("ml/knn_new_tfidf_vectorizer.joblib")
data = pd.read_csv(r"/Users/alenaagafonova/Downloads/new_clean_data (7).csv")


class UserInput(BaseModel):
    id: int
    interests: str
    intents: str

@app.get("/me", summary="read me",
         description="Следующий endpoint принимает входные данные пользователя и "
                     "предоставляет рекомендации на основе его интересов и намерений, "
                     "используя алгоритм K-ближайших соседей (KNN).")
def read_me():
    return {"message": "Welcome to the KNN Recommendation API"}


@app.post("/get-recommendations/", summary="recommendations for exhibitors")
def get_recommendations(user_id: int):
    # Создание пар интересов только однажды
    data_pairs = []
    for _, row in data.iterrows():
        interests_company = row['Interests_scanned_by']
        interests_visitor = row['Interests_was_scanned']
        data_pairs.append(interests_company + " | " + interests_visitor)
    # Преобразование текстовых данных в числовой формат с использованием загруженного векторизатора
    X = vectorizer.transform(data_pairs)
    # Найти индекс пользователя по id_scanned_by
    index = data[data['id_scanned_by'] == user_id].index
    if not index.empty:
        user_vector = X[index][0]  # Получаем эмбеддинг пользователя
        distances, indices = model.kneighbors(user_vector, n_neighbors=5)  # Находим ближайших соседей
        # Извлекаем рекомендации
        recommendations = data.iloc[indices.flatten()]
        recommendations = recommendations[recommendations['id_was_scanned'] != user_id]  # Убираем текущего пользователя
        unique_recommendations = recommendations.drop_duplicates(subset='id_was_scanned')
        # Формируем ответ
        result = unique_recommendations[['id_was_scanned', 'Interests_was_scanned','Intents_was_scanned']].to_dict(orient='records')
        if result:
            return {"user_id": user_id, "recommendations": result}
        else:
            raise HTTPException(status_code=404, detail="Рекомендации не найдены")
    else:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
