import json
import requests

BASE_URL = "http://127.0.0.1:5000/"

# You can use this function to test your api
# Make sure the server is running locally on BASE_URL`
# or change the hardcoded localhost above
def test_predict():
    """
    Test the predict route with test data
    """
    test_description = {
        "description": "A Phase III clinical trial to evaluate the efficacy of a novel therapeutic intervention in patients with early-stage Alzheimer's disease and other forms of dementia. The study will assess cognitive function, daily living activities, and biomarkers of neurodegeneration over a 24-month period."
    }
    headers = {'Content-Type': 'application/json'}
    
    print("Calling API with test description:")
    print(f"Data: {json.dumps(test_description, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=test_description,  # Using json parameter instead of data for automatic serialization
        headers=headers
    )
    
    print("\nResponse:")
    print(f"Status Code: {response.status_code}")
    
    try:
        print(f"Response JSON: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        print(f"Raw response: {response.text}")
    
    assert response.status_code == 200


if __name__ == "__main__":
    test_predict()