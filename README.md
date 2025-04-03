# Medical Trial Text Classification

This project implements a medical trial text classifier using PubMedBERT to categorize clinical trial descriptions into five disease categories.

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
cd src
python train.py
```

3. Evaluate the model:
```bash
python evaluate.py
```
This will:
- Generate a classification report with precision, recall, and F1 scores
- Create a confusion matrix visualization (saved as confusion_matrix.png)
- Evaluate the model on the test dataset

4. Start the Flask server:
```bash
python main.py
```

5. Test the API:
```bash
python test.py
```

## Design Choices

### Model Selection
- **PubMedBERT**: Chosen because it's specifically pre-trained on biomedical text, making it ideal for understanding medical trial descriptions
- **Architecture**: Fine-tuned transformer with a classification head
- **Input Processing**: Truncated to 512 tokens with proper padding

### Training Strategy
- 80-20 train-validation split
- Stratified sampling to handle class balance
- Early stopping based on validation loss
- Different learning rates for BERT and classifier layers
- Batch size of 8 to balance memory usage and training speed

### API Design
- RESTful endpoint for predictions
- JSON input/output
- Proper error handling
- Stateless design for scalability

## API Usage

Send POST requests to `/predict` endpoint:

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"description": "A clinical trial studying cognitive decline in elderly patients..."}'
```

Response format:
```json
{
    "prediction": "Dementia"
}
```

## Model Performance

The model is evaluated on a held-out validation set with metrics including:
- Accuracy
- Loss curves
- Per-class precision/recall

## Future Improvements

1. Add cross-validation for more robust evaluation
2. Implement model versioning
3. Add confidence scores to predictions
4. Add more extensive error handling and input validation
5. Implement batch prediction endpoint for multiple descriptions

## Project Structure

```
.
├── data/
│   └── trials.csv
├── src/
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── main.py
├── test.py
├── requirements.txt
└── README.md
```
