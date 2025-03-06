# ((!!WRITEN BY CLAUDE!!)) Google Cloud Translation API - Simple Module

This document explains how to set up and use the simplified Google Cloud Translation API module.

## Prerequisites

1. A Google Cloud account with billing enabled
2. A Google Cloud project with the Translation API enabled
3. An API key with access to the Translation API

## Setup Instructions

### 1. Install the required Python packages

```bash
pip install requests python-dotenv
```

### 2. Create an API Key

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project
3. Navigate to "APIs & Services" > "Credentials"
4. Click "Create Credentials" > "API key"
5. Copy your new API key
6. (Optional but recommended) Restrict the API key to only the Translation API

### 3. Set Up Your API Key

Create a `.env` file in the root directory of your project with the following content:

```
GoogleCloudAPIKEY = "your-actual-api-key"
```

## Usage

The module is designed to be simple and focused on the core translation functionality.

### Basic Usage

```python
from Scraping.example_cloud import translate_sentences

# Translate a single sentence
result = translate_sentences("Hello, how are you?", target_language="fr", source_language="en")
print(result)  # "Bonjour, comment allez-vous?"

# Translate multiple sentences
sentences = ["Hello", "How are you?", "Nice to meet you"]
results = translate_sentences(sentences, target_language="es", source_language="en")
print(results)  # ["Hola", "¿Cómo estás?", "Encantado de conocerte"]
```

### Working with Files

The module doesn't handle files directly, but you can easily add file handling:

```python
# Read from file
with open("input.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

# Translate
translations = translate_sentences(sentences, target_language="fr")

# Write to file
with open("output.txt", "w", encoding="utf-8") as f:
    for translation in translations:
        f.write(f"{translation}\n")
```

For convenience, we've included a file handling example in `translate_file_example.py`.

## Supported Languages

For a complete list of supported languages and their codes, visit:
https://cloud.google.com/translate/docs/languages

Common language codes:
- English: `en`
- French: `fr`
- Spanish: `es`
- German: `de`
- Chinese (Simplified): `zh-CN`
- Japanese: `ja`
- Dzongkha: `dz`

## Troubleshooting

If you encounter errors:

1. Verify that your API key is correct in the .env file
2. Check that the Translation API is enabled in your Google Cloud project
3. Ensure your Google Cloud project has billing enabled
4. Make sure the .env file is in the correct location (root of your project)

## API Key Security

Storing API keys in a .env file is more secure than hardcoding them in your scripts, but for production applications, consider using a more secure method like environment variables or a secret management service. 