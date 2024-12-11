import unittest
from app import app

class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_home_route(self):
        # Test the home route (if applicable)
        response = self.client.get('/')
        self.assertEqual(response.status_code, 404)  # Update based on your home route existence

    def test_model_prediction_get(self):
        # Test the GET request for the model_prediction route
        response = self.client.get('/model_prediction')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Model Prediction', response.data)

    def test_model_prediction_post(self):
        # Test POST request for model_prediction route
        form_data = {
            "longitude": "-122.23",
            "latitude": "37.88",
            "housing_median_age": "41",
            "total_rooms": "880",
            "total_bedrooms": "129",
            "population": "322",
            "households": "126",
            "median_income": "8.3252",
            "ocean_proximity_NEAR_BAY": "1",
            "ocean_proximity_INLAND": "0",
            "ocean_proximity_NEAR_OCEAN": "0",
            "ocean_proximity_ISLAND": "0"
        }
        response = self.client.post('/model_prediction', data=form_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Predicted House Price', response.data)

    def test_visualization_route(self):
        # Test the visualization route
        response = self.client.get('/visualization')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Data Visualizations', response.data)

    def test_descriptive_stats_route(self):
        # Test the descriptive_stats route
        response = self.client.get('/descriptive_stats')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Descriptive Statistics', response.data)

if __name__ == "__main__":
    unittest.main()
