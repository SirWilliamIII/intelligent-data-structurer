# 🧠 Intelligent Data Processor

**Transform unstructured data into organized, searchable MongoDB collections automatically using AI-powered classification.**

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=flat&logo=mongodb&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- 🤖 **AI-Powered Classification** - Advanced NLP using spaCy for intelligent content analysis
- 🗂️ **Dynamic Collections** - Automatically creates MongoDB collections based on content patterns
- 📊 **Confidence Scoring** - Quality assurance with visual confidence indicators
- 🔄 **Learning System** - Improves classification accuracy over time through similarity matching
- 🌐 **Modern Web UI** - Beautiful drag-and-drop interface built with Alpine.js and Tailwind CSS
- 📁 **Multi-Format Support** - Process text, markdown, PDFs, images, and more
- 🎯 **Schema Evolution** - Intelligent database schema suggestions based on content patterns

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- MongoDB running on localhost:27017
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/intelligent-data-processor.git
cd intelligent-data-processor
```

2. **Install dependencies**
```bash
uv sync
```

3. **Download spaCy language model**
```bash
uv run python -m spacy download en_core_web_sm
```

4. **Start MongoDB** (if not already running)
```bash
brew services start mongodb/brew/mongodb-community
# or
mongod
```

5. **Run the application**
```bash
uv run python main.py
```

6. **Open your browser**
Navigate to `http://localhost:8000`

## 🎯 How It Works

### 1. **Upload Files**
Drag and drop or select files through the web interface. Supports:
- Text files (.txt, .md, .log)
- Documents (.pdf, .docx)
- Images (.png, .jpg) with OCR
- Structured data (.csv, .json)

### 2. **AI Analysis**
The system analyzes content using multiple strategies:
- **Semantic Signature Analysis** - Extracts domain keywords and structural patterns
- **NLP Entity Recognition** - Identifies people, organizations, technologies
- **Similarity Matching** - Compares against previously processed content
- **Confidence Scoring** - Provides quality metrics for each classification

### 3. **Dynamic Collection Creation**
Based on analysis, creates MongoDB collections with meaningful names:
- `kubernetes_resources` - For Kubernetes documentation
- `spanish_language` - For Spanish learning materials
- `football_sports` - For sports-related content
- `cooking_recipes` - For recipe content
- And many more, generated dynamically!

### 4. **Intelligent Storage**
Documents are stored with rich metadata including:
- Original content and cleaned version
- Extracted entities and keywords
- Classification confidence and reasoning
- Semantic fingerprint for future similarity matching

## 🏗️ Architecture

```
intelligent-data-processor/
├── core/
│   ├── intelligent_analyzer.py    # Main AI analysis engine
│   ├── classifier.py             # Content classification logic
│   ├── mongo_database.py         # MongoDB management
│   ├── database.py              # Database abstraction layer
│   ├── document_processor.py    # File processing utilities
│   └── config.py               # Configuration management
├── templates/
│   └── index.html              # Web interface
├── main.py                     # FastAPI application
└── pyproject.toml             # Dependencies and metadata
```

## 🔧 Configuration

Create a `.env` file for custom configuration:

```env
# MongoDB Settings
MONGO_URL=mongodb://localhost:27017
DATABASE_NAME=intelligent_data

# AI Settings
CONFIDENCE_THRESHOLD=0.7
SPACY_MODEL=en_core_web_sm

# Application Settings
DEBUG=true
LOG_LEVEL=INFO
```

## 📊 Example Classifications

The system intelligently categorizes content:

| File Type | Example | Collection | Confidence |
|-----------|---------|------------|------------|
| Kubernetes Docs | `kubectl-cheatsheet.md` | `kubernetes_resources` | 95% |
| Sports List | `sports.txt` | `football_sports` | 80% |
| Recipe | `chocolate-cake.md` | `cooking_recipes` | 88% |
| Technical Ref | `linux-commands.txt` | `linux_technical` | 92% |

## 🛠️ API Endpoints

### Classification Only
```bash
curl -X POST -F "file=@document.txt" http://localhost:8000/classify
```

### Process and Store
```bash
curl -X POST -F "file=@document.txt" http://localhost:8000/process
```

### Health Check
```bash
curl http://localhost:8000/health
```

### List Collections
```bash
curl http://localhost:8000/database/tables
```

## 🧪 Development

### Running Tests
```bash
uv run pytest
```

### Code Formatting
```bash
uv run black .
uv run isort .
```

### Test Classification
```bash
uv run python test_classification.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Dependencies

- **FastAPI** - Modern web framework for the API
- **MongoDB/Motor** - Database and async driver
- **spaCy** - Natural language processing
- **Loguru** - Structured logging
- **Pydantic** - Data validation
- **Alpine.js** - Reactive web components
- **Tailwind CSS** - Utility-first styling

## 🔮 Future Enhancements

- [ ] Support for more file formats (Excel, PowerPoint)
- [ ] Advanced similarity search with vector embeddings
- [ ] Real-time collaboration features
- [ ] Export capabilities (JSON, CSV)
- [ ] Custom classification rules
- [ ] Integration with cloud storage (S3, Google Drive)
- [ ] REST API for programmatic access
- [ ] Batch processing capabilities

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with modern Python tools: [uv](https://github.com/astral-sh/uv), FastAPI, and MongoDB
- Powered by [spaCy](https://spacy.io/) for natural language processing
- UI inspired by modern design principles with Tailwind CSS

---

**Transform your unstructured data into organized knowledge with AI! 🚀**