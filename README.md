# Intelligent Data Processor

An advanced system that intelligently processes unstructured data (text, markdown, images) and automatically creates structured database tables based on content classification.

## Features

🧠 **Intelligent Classification** - Automatically detects content types:
- Contact Information & Business Cards
- Product Data & Invoices  
- Event Information & Meeting Notes
- Articles & Blog Posts
- Recipes with Ingredients
- Financial Transactions
- Log Entries & System Data
- Employee & HR Data
- Email Threads

🗄️ **Dynamic Schema Creation** - Creates appropriate database tables:
- `contacts` - Names, emails, phones, addresses
- `products` - SKUs, prices, inventory
- `events` - Dates, times, locations
- `articles` - Titles, authors, content
- `recipes` + `ingredients` + `instructions`
- `transactions` - Financial data
- `logs` - System monitoring
- `employees` - HR information

⚡ **Background Processing** - Asynchronous job queue with progress tracking

🔍 **Confidence Scoring** - Smart confidence-based workflow:
- Auto-process high confidence (>0.8)
- Queue medium confidence for review (0.4-0.8)  
- Flag low confidence for manual classification (<0.4)

🖼️ **Image Processing** - OCR support for extracting text from images

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL
- Redis
- Tesseract OCR (optional, for image processing)

### Installation

1. **Clone and setup:**
```bash
cd ~/Programming
git clone <repository> intelligent-data-processor
cd intelligent-data-processor
python setup.py
```

2. **Activate environment:**
```bash
source venv/bin/activate
```

3. **Configure settings:**
```bash
cp .env.example .env
# Edit .env with your database and Redis settings
```

4. **Run the application:**
```bash
python main.py
```

5. **Visit:** `http://localhost:8000`

### Manual Setup

If automatic setup fails:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Setup database
createdb intelligent_data

# Start services
brew services start postgresql  # macOS
brew services start redis       # macOS
```

## Usage

### Upload Files

1. Go to `http://localhost:8000`
2. Upload text files, markdown, or images
3. System automatically:
   - Classifies content type
   - Extracts structured data
   - Creates appropriate database tables
   - Stores data with confidence scores

### Monitor Processing

- **High confidence** → Auto-processed immediately
- **Medium confidence** → Queued for review
- **Low confidence** → Manual classification needed

### Query Data

```bash
# View all tables created
python -c "from core.database import list_tables; print(list_tables())"

# Query specific data
psql intelligent_data -c "SELECT * FROM contacts;"
psql intelligent_data -c "SELECT * FROM products;"
```

## Architecture

```
Upload → Classifier → Confidence Score → Background Job → Extractor → Validator → Database
```

### Content Types Supported

| Type | Auto-Creates | Fields |
|------|-------------|--------|
| Contacts | `contacts` | name, email, phone, address, company |
| Products | `products` | name, sku, price, category, inventory |
| Events | `events` | title, date, time, location, organizer |
| Articles | `articles` | title, author, content, tags |
| Recipes | `recipes`, `ingredients`, `instructions` | Relational recipe data |
| Financial | `transactions` | date, amount, account, type |
| Logs | `logs` | timestamp, level, service, message |
| Employees | `employees` | id, name, department, salary |

## Configuration

Key settings in `.env`:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/intelligent_data

# Redis (for background jobs)
REDIS_URL=redis://localhost:6379/0

# AI/ML
CONFIDENCE_THRESHOLD=0.7
SPACY_MODEL=en_core_web_sm

# File Processing
MAX_FILE_SIZE=50MB
UPLOAD_DIR=./uploads
```

## Development

### Project Structure

```
intelligent-data-processor/
├── app/              # FastAPI application
├── core/             # Core processing logic
│   ├── classifier.py # Content classification
│   ├── extractors.py # Data extraction
│   ├── database.py   # Database management
│   └── jobs.py       # Background jobs
├── api/              # API endpoints
├── templates/        # Web interface
├── tests/            # Test cases
└── uploads/          # File storage
```

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black .
isort .
```

## Examples

### Contact Information
```
John Smith
CEO, Tech Corp
john@techcorp.com
(555) 123-4567
123 Main St, Seattle, WA
```
→ Creates record in `contacts` table

### Product Data
```
Product: Wireless Headphones
SKU: WH-1000XM5
Price: $299.99
Category: Electronics
In Stock: 45 units
```
→ Creates record in `products` table

### Recipe
```
# Chocolate Chip Cookies

Ingredients:
- 2 cups flour
- 1 cup butter
- 1/2 cup sugar

Instructions:
1. Preheat oven to 350°F
2. Mix ingredients
3. Bake for 12 minutes
```
→ Creates records in `recipes`, `ingredients`, and `instructions` tables

## Troubleshooting

### Database Connection Issues
```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Recreate database
dropdb intelligent_data
createdb intelligent_data
```

### Redis Connection Issues
```bash
# Check Redis is running
redis-cli ping

# Start Redis
brew services start redis  # macOS
```

### spaCy Model Issues
```bash
# Download model manually
python -m spacy download en_core_web_sm
```

## License

MIT License
