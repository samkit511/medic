# Medical Chatbot - Setup Guide

This guide will help you set up the Medical Chatbot application on your local system with all required dependencies, including AI/ML models and testing frameworks.

## 📋 Prerequisites

- Python 3.8+ (Python 3.11+ recommended)
- Node.js 16+ (for frontend)
- Git
- At least 4GB RAM (8GB recommended for AI models)
- 10GB+ free disk space (for AI model downloads)

## 🚀 Quick Setup

### 1. Clone and Navigate

```bash
git clone <your-repo-url>
cd medic
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv medic_env

# Activate virtual environment
# Windows:
medic_env\Scripts\activate
# Linux/Mac:
source medic_env/bin/activate
```

### 3. Install Python Dependencies

```bash
# Install all Python dependencies
pip install -r requirements.txt
```

This will install:
- ✅ **Core API Framework** (FastAPI, Uvicorn, Pydantic)
- ✅ **AI/ML Libraries** (Transformers, PyTorch, Hugging Face)
- ✅ **Security & Encryption** (Cryptography, JWT, Google Auth)
- ✅ **Database** (SQLite with encryption)
- ✅ **Testing Framework** (Pytest with all extensions)
- ✅ **Medical AI Models** (BioBERT, medical NER)
- ✅ **Document Processing** (PDF, OCR, Word docs)

### 4. Install Frontend Dependencies

```bash
npm install
```

### 5. Verify Installation

```bash
# Run dependency verification
python verify_dependencies.py
```

This script will check:
- ✅ Python version compatibility
- ✅ All core dependencies
- ✅ AI/ML libraries
- ✅ Security components
- ✅ Testing framework
- ✅ System requirements

## 🏥 Medical AI Models Setup

The application uses several medical AI models that will be downloaded automatically:

### Core Models (Auto-downloaded on first use):
- **sentence-transformers/all-MiniLM-L6-v2** - Text embeddings
- **microsoft/DialoGPT-medium** - Conversational AI
- **BioBERT** - Medical text understanding
- **Medical NER** - Named entity recognition

### Model Storage:
Models are cached in:
- `models_cache/` directory (ignored by git)
- Hugging Face cache directory (`~/.cache/huggingface/`)

**First run will take 5-10 minutes** to download models.

## 🔐 Environment Configuration

### 1. Create Environment File

```bash
cp .env.example .env  # If .env.example exists
# or create .env manually
```

### 2. Configure API Keys

Edit `.env` file:

```env
# AI API Keys
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here

# Security
JWT_SECRET_KEY=your_very_secure_jwt_secret_key_here
DATABASE_ENCRYPTION_KEY=your_database_encryption_key_here

# Google OAuth (Optional)
GOOGLE_CLIENT_ID=your_google_client_id_here

# App Configuration
DEBUG=True
LOG_LEVEL=INFO
```

**🔒 Security Note:** Never commit the `.env` file! It's already in `.gitignore`.

## 🧪 Testing Setup

### Run Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m api           # API tests only
pytest -m security      # Security tests only

# Run with coverage
pytest --cov=backend --cov-report=html
```

### Test Structure:
- `tests/` - All test files
- `tests/conftest.py` - Shared fixtures and configuration
- `tests/test_api.py` - API endpoint tests
- `pytest.ini` - Pytest configuration

## 🏃‍♂️ Running the Application

### 1. Start Backend (Python)

```bash
cd backend
python main.py
```

Backend will run on:
- HTTP: `http://localhost:8000`
- HTTPS: `https://localhost:8443` (if SSL configured)

### 2. Start Frontend (Node.js)

```bash
# In another terminal
npm run dev
```

Frontend will run on:
- `http://localhost:5173` (Vite default)

## 📊 System Monitoring

### Health Check Endpoints:
- `GET /` - Basic API information
- `GET /health` - System health status
- `GET /docs` - API documentation (Swagger)

### Logs:
- Backend logs in `logs/` directory
- HIPAA audit logs (kept for compliance)

## 🔧 Development Tools

### Code Quality:
```bash
# Format code
black backend/

# Sort imports
isort backend/

# Lint code
flake8 backend/

# Type checking
mypy backend/
```

### Database:
- SQLite database with encryption
- Location: `data/chatbot.db`
- Encryption key: `data/db_key.key`

## 📚 Additional Setup

### Tesseract OCR (for image processing):

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### GPU Support (Optional):
For faster AI inference, install CUDA toolkit:
- Download from: https://developer.nvidia.com/cuda-downloads
- PyTorch will automatically detect and use GPU

## 🐳 Docker Setup (Alternative)

```dockerfile
# Create Dockerfile if needed
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "backend/main.py"]
```

## 📈 Performance Optimization

### Recommended System Specs:
- **Minimum:** 4GB RAM, 10GB storage
- **Recommended:** 8GB+ RAM, 20GB+ storage, GPU
- **Production:** 16GB+ RAM, SSD storage, dedicated GPU

### Model Optimization:
- Models are cached after first download
- Use `model_offload/` for memory management
- Consider quantized models for lower memory usage

## 🆘 Troubleshooting

### Common Issues:

1. **Import Errors:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

2. **Memory Issues:**
   - Reduce concurrent users
   - Use smaller AI models
   - Increase system RAM

3. **Model Download Fails:**
   - Check internet connection
   - Clear model cache: `rm -rf models_cache/`
   - Try manual download

4. **SSL Certificate Issues:**
   - Run `python backend/trust_certificate.ps1` (Windows)
   - Or disable SSL in development

### Get Help:
- Check `verify_dependencies.py` output
- Review logs in `logs/` directory  
- Run tests to identify issues: `pytest -v`

## 📝 Next Steps

1. ✅ Verify all dependencies are installed
2. ✅ Configure environment variables
3. ✅ Run tests to ensure everything works
4. ✅ Start the application
5. ✅ Test medical queries
6. ✅ Set up production deployment

---

## 🔒 HIPAA Compliance Notes

This application includes HIPAA compliance features:
- Database encryption
- Audit logging
- Secure authentication
- PHI data protection

**⚠️ Important:** For production use, ensure:
- Strong encryption keys
- Regular security audits  
- Proper access controls
- Compliance with local regulations

---

**Need help?** Check the troubleshooting section or run `python verify_dependencies.py` for detailed diagnostics.
