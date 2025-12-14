"""
API Tests for ChestX-AI
"""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np


# Create a simple test image
def create_test_image(size=(224, 224)):
    """Create a dummy grayscale image for testing"""
    img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self, client):
        """Test that health endpoint returns healthy status"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "ChestX-AI"


class TestDiseasesEndpoint:
    """Test diseases list endpoint"""
    
    def test_list_diseases(self, client):
        """Test that diseases endpoint returns all 14 diseases"""
        response = client.get("/api/diseases")
        assert response.status_code == 200
        
        data = response.json()
        assert data["count"] == 14
        assert len(data["diseases"]) == 14
        
        # Check disease structure
        disease = data["diseases"][0]
        assert "name" in disease
        assert "description" in disease


class TestPredictEndpoint:
    """Test prediction endpoint"""
    
    def test_predict_valid_image(self, client):
        """Test prediction with valid image"""
        image = create_test_image()
        
        response = client.post(
            "/api/predict",
            files={"file": ("test.png", image, "image/png")}
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert len(data["predictions"]) == 14
        assert "heatmap" in data
        assert "inference_time_ms" in data
        assert "device" in data
    
    def test_predict_invalid_file_type(self, client):
        """Test prediction with invalid file type"""
        response = client.post(
            "/api/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]
    
    def test_predict_no_file(self, client):
        """Test prediction without file"""
        response = client.post("/api/predict")
        assert response.status_code == 422


class TestPredictionResults:
    """Test prediction result structure"""
    
    def test_prediction_structure(self, client):
        """Test that predictions have correct structure"""
        image = create_test_image()
        
        response = client.post(
            "/api/predict",
            files={"file": ("test.png", image, "image/png")}
        )
        
        data = response.json()
        prediction = data["predictions"][0]
        
        assert "disease" in prediction
        assert "probability" in prediction
        assert "severity" in prediction
        
        # Probability should be between 0 and 1
        assert 0 <= prediction["probability"] <= 1
        
        # Severity should be valid
        assert prediction["severity"] in ["low", "medium", "high"]
    
    def test_predictions_sorted(self, client):
        """Test that predictions are sorted by probability"""
        image = create_test_image()
        
        response = client.post(
            "/api/predict",
            files={"file": ("test.png", image, "image/png")}
        )
        
        data = response.json()
        probs = [p["probability"] for p in data["predictions"]]
        
        # Should be sorted in descending order
        assert probs == sorted(probs, reverse=True)


# Pytest fixtures
@pytest.fixture
def client():
    """Create test client"""
    from app.main import app
    return TestClient(app)
