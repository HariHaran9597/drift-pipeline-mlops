# Drift-Pipeline: Self-Healing ML System

A production-ready MLOps platform that automatically detects data drift and retrains machine learning models in response to distribution shifts. Built with FastAPI, PyTorch, and Prefect for robust time-series demand forecasting.

## Overview

Drift-Pipeline is an end-to-end machine learning system designed to maintain model accuracy in production by:

- **Detecting Data Drift**: Uses Evidently AI for statistical analysis of data distribution changes
- **Automatic Retraining**: Orchestrates model retraining workflows via Prefect when drift is detected
- **Real-time Predictions**: Serves predictions through a FastAPI REST endpoint
- **Comprehensive Monitoring**: Integrates Prometheus and Grafana for system observability

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ML Application Stack                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   FastAPI    │  │   Prefect    │  │   Training   │      │
│  │   (Serving)  │  │ (Orchestr.)  │  │   (LSTM)     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                            │                                 │
│                   ┌────────▼────────┐                        │
│                   │   PostgreSQL    │                        │
│                   │  (Feature Store)│                        │
│                   └─────────────────┘                        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
    ┌────▼────┐        ┌──────▼───────┐    ┌──────▼──────┐
    │Prometheus│       │   Grafana    │    │Drift Monitor│
    │(Metrics) │       │  (Dashboard) │    │  (Alerts)   │
    └──────────┘       └──────────────┘    └─────────────┘
```

### Technology Stack

| Component | Technology |
|-----------|------------|
| **Web Framework** | FastAPI + Uvicorn |
| **Deep Learning** | PyTorch (LSTM) |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Drift Detection** | Evidently AI |
| **Orchestration** | Prefect v2 |
| **Database** | PostgreSQL |
| **ORM** | SQLAlchemy |
| **Monitoring** | Prometheus + Grafana |
| **Containerization** | Docker & Docker Compose |

## Features

✨ **Automatic Drift Detection**
- Statistical tests on historical vs. current data
- Configurable baseline and detection windows
- Detailed drift reports saved as JSON

⚡ **Self-Healing Model Pipeline**
- Automatic retraining triggered on drift detection
- LSTM model for time-series forecasting
- MinMax scaling for feature normalization

📊 **Real-time API**
- REST endpoint for demand predictions
- Accepts weather features (temperature, humidity)
- 30-day lookback window for context

📈 **Observability**
- Prometheus metrics collection
- Grafana dashboards for visualization
- FastAPI health endpoints

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.10+
- PostgreSQL (or use containerized version)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd drift-pipeline
```

2. **Start the stack**
```bash
docker-compose up -d
```

This starts:
- PostgreSQL on `localhost:5432`
- ML App with FastAPI on `localhost:8000`
- Prometheus on `localhost:9090`
- Grafana on `localhost:3000`

3. **Populate the database**
```bash
docker exec drift_ml_app python scripts/populate_db.py
```

Generates 2 years of synthetic weather and demand data.

### Usage

#### Make Predictions

```bash
python test_api.py
```

Or with curl:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"temperature": 35.5, "humidity": 40.0}'
```

Response:
```json
{
  "model_version": "v1",
  "predicted_demand": 245.32
}
```

#### Run Drift Detection & Retraining

```bash
docker exec drift_ml_app python -m src.orchestration.flow
```

Or manually check for drift:
```bash
docker exec drift_ml_app python -m src.drift.monitor
```

#### Generate Traffic (Load Testing)

```bash
python scripts/generate_traffic.py
```

Continuously sends prediction requests to monitor system behavior.

#### View Dashboards

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (default: admin/admin)

## Project Structure

```
drift-pipeline/
├── src/
│   ├── serving/
│   │   └── api.py                 # FastAPI application
│   ├── drift/
│   │   └── monitor.py             # Drift detection logic
│   ├── training/
│   │   └── train.py               # Model retraining pipeline
│   ├── models/
│   │   ├── lstm.py                # LSTM architecture
│   │   ├── production_model.pt    # Trained model weights
│   │   └── scaler.pkl             # Feature scaler
│   ├── database/
│   │   └── db.py                  # Database utilities
│   └── orchestration/
│       └── flow.py                # Prefect workflow
├── scripts/
│   ├── populate_db.py             # Synthetic data generation
│   ├── generate_traffic.py        # Load testing script
│   └── init.sql                   # Database schema
├── monitoring/
│   ├── prometheus.yml             # Prometheus config
│   └── grafana_dashboard.json     # Grafana dashboard
├── data/
│   ├── raw/                       # Raw data
│   ├── processed/                 # Processed data
│   ├── reference/                 # Reference datasets
│   ├── models/                    # Model artifacts
│   └── drift_report.json          # Latest drift report
├── config/                        # Configuration files
├── docker-compose.yml             # Service orchestration
├── Dockerfile                     # Container image
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## API Documentation

### Endpoints

#### `POST /predict`

Generates demand prediction based on weather features.

**Request:**
```json
{
  "temperature": 25.5,
  "humidity": 65.0
}
```

**Response (200 OK):**
```json
{
  "model_version": "v1",
  "predicted_demand": 234.56
}
```

**Errors:**
- `500`: Insufficient historical data or prediction error

#### Metrics

FastAPI Instrumentator automatically exposes Prometheus metrics at `/metrics`.

## Configuration

### Environment Variables

Set in `docker-compose.yml`:

```yaml
DATABASE_URL=postgresql://user:password@postgres:5432/feature_store
```

### Model Configuration

Edit in `src/training/train.py`:
- `LOOKBACK_WINDOW`: Sequence length for LSTM (default: 30 days)
- `EPOCHS`: Training epochs (default: 20)
- `HIDDEN_SIZE`: LSTM hidden units (default: 50)

### Drift Detection Configuration

Edit in `src/drift/monitor.py`:
- Reference window: First 500 records
- Current window: Last 30 records
- Test method: Evidently AI DataDriftPreset

## Model Details

### LSTM Architecture

```
Input (batch_size, 30, 2) 
  ↓
LSTM Layer (50 hidden units, batch_first=True)
  ↓
Fully Connected Layer (50 → 1)
  ↓
Output (batch_size, 1) - Demand prediction
```

### Training Pipeline

1. **Load**: All historical data from PostgreSQL
2. **Scale**: MinMax scaling (0-1 normalization)
3. **Sequence Creation**: 30-day windows with next-day target
4. **Train**: 20 epochs with Adam optimizer (lr=0.01)
5. **Evaluate**: Calculate RMSE on full dataset
6. **Save**: Model weights and scaler

### Input Features

- Temperature (°C): 15-35 range
- Humidity (%): 30-90 range

### Target Variable

- Demand: Correlated with temperature and humidity

## Monitoring & Observability

### Prometheus Metrics

The system exposes:
- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request latency
- Custom metrics via FastAPI Instrumentator

### Grafana Dashboards

Create dashboards to visualize:
- Prediction request rate
- Model inference latency
- System resource usage
- Drift detection status

## Troubleshooting

### Issue: Connection refused to PostgreSQL

**Solution**: Ensure PostgreSQL container is running
```bash
docker ps | grep postgres
```

### Issue: Model not loading

**Solution**: Run training script to generate model artifacts
```bash
docker exec drift_ml_app python src/training/train.py
```

### Issue: Drift detection fails

**Solution**: Ensure sufficient data in database (>530 records)
```bash
docker exec drift_ml_app python scripts/populate_db.py
```

## Performance

### Benchmarks

- **Prediction Latency**: ~5-10ms per request
- **Model Training**: ~30-60s for 2 years of data
- **Drift Detection**: ~2-5s statistical analysis

### Scalability

- Docker containers horizontally scalable
- PostgreSQL connection pooling via SQLAlchemy
- Async API endpoints via FastAPI/Uvicorn

## Development

### Running Locally

Without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Set database URL
export DATABASE_URL=postgresql://user:password@localhost:5432/feature_store

# Start API
python -m uvicorn src.serving.api:app --reload --port 8000
```

### Testing

```bash
# Unit tests (if available)
pytest tests/

# Integration test
python test_api.py
```

## Deployment

### Production Checklist

- [ ] Configure PostgreSQL with proper backups
- [ ] Set environment variables securely
- [ ] Enable SSL/TLS for API endpoints
- [ ] Configure Grafana authentication
- [ ] Set up alert thresholds in Prometheus
- [ ] Implement model versioning strategy
- [ ] Set up CI/CD pipeline
- [ ] Configure resource limits in docker-compose

### Docker Registry

```bash
docker build -t your-registry/drift-pipeline:latest .
docker push your-registry/drift-pipeline:latest
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact the development team
- Check the documentation wiki

## Roadmap

- [ ] Multi-model ensemble support
- [ ] Advanced drift metrics (KL divergence, Wasserstein)
- [ ] A/B testing framework for model versions
- [ ] Real-time feature importance tracking
- [ ] Automated hyperparameter tuning
- [ ] MLflow integration for experiment tracking
- [ ] Kubernetes deployment templates

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern async web framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Evidently AI](https://www.evidentlyai.com/) - ML monitoring
- [Prefect](https://www.prefect.io/) - Workflow orchestration
- [PostgreSQL](https://www.postgresql.org/) - Relational database

---

**Built with ❤️ for ML Operations**
