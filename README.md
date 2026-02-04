# TaskEval

## Setup

### Prerequisites

1. **Python 3.10+** - Make sure you have Python 3.10 or higher installed
2. **Poetry** - For dependency management
3. **direnv** - For environment variable management

### Installation

1. **Install direnv** (if not already installed):

   ```bash
   # On macOS with Homebrew
   brew install direnv

   # On Ubuntu/Debian
   sudo apt-get install direnv

   # On other systems, check: https://direnv.net/docs/installation.html
   ```

2. **Install Python dependencies**:

   ```bash
   poetry install
   ```

3. **Set up environment variables**:

   - Copy the example environment file:

   ```bash
   cp .envrc.example .envrc
   ```

   - Edit `.envrc` and add your API keys:

   ```bash
   export ANTHROPIC_API_KEY="your-actual-api-key-here"
   export OPENAI_API_KEY="your-actual-api-key-here"
   ```

   Note: You'll need to obtain API keys from:
   - Anthropic API: https://console.anthropic.com/
   - OpenAI API: https://platform.openai.com/api-keys

4. **Allow direnv to load the environment**:

   ```bash
   direnv allow
   ```

### Usage

## Running the Application

### Start the Streamlit App

To run the TaskEval Streamlit application:

```bash
# Activate the Poetry environment and run Streamlit
poetry run streamlit run app.py
```

Or if you prefer to activate the virtual environment first:

```bash
# Activate the Poetry shell
poetry shell

# Run the Streamlit app
streamlit run app.py
```

The application will start and be available at `http://localhost:8501` in your web browser.

### Application Features

The TaskEval app provides two main evaluation tasks:

1. **Figure Extraction**: Upload an image and provide instructions to extract data points from charts/graphs
2. **QA Answering**: Upload a PDF document and ask questions about its content

### Troubleshooting

If you encounter any issues:

1. **Port already in use**: If port 8501 is busy, Streamlit will automatically use the next available port (8502, 8503, etc.)

2. **Environment variables not loaded**: Make sure you've run `direnv allow` in the project directory

3. **Dependencies not found**: Ensure you've run `poetry install` and activated the environment with `poetry shell`

4. **API key issues**: Verify your API keys are correctly set in the `.envrc` file
