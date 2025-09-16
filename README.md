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

   - Edit `.envrc` and replace the placeholder API key with your actual Anthropic API key:

   ```bash
   export ANTHROPIC_API_KEY="your-actual-api-key-here"
   ```

4. **Allow direnv to load the environment**:

   ```bash
   direnv allow
   ```

### Usage

The tool can be run in two ways:

#### Method 1: Using Poetry (Recommended)

```bash
poetry run python -m taskevals.pipeline
```

#### Method 2: Using Python directly (after activating the environment)

```bash
poetry shell
python -m taskevals.pipeline
```

#### Command Line Arguments

The tool accepts the following parameters:

- `--task`: The task description (default: "Extract the data points in the chart image and provide the output as a table.")
- `--task_input`: Path to the input file (default: "./new_chart_images/1.png")
- `--llm_output`: Path to the output file (default: "./chart_xlxs_outputs/1.xlsx")

#### Example Usage

```bash
# Run with default parameters
poetry run python -m taskevals.pipeline

# Run with custom parameters
poetry run python -m taskevals.pipeline \
  --task "Extract the data points in the chart image and provide the output as a table." \
  --task_input "./data/new_chart_images/1.png" \
  --llm_output "./outputs/1.xlsx"
```

### Output

The tool generates Excel files with the following structure:

- **Output sheet**: Contains the generated data in a single row
- **Scenario_Info sheet**: Contains test scenario metadata

### Troubleshooting

1. **API Key Issues**: Make sure your `ANTHROPIC_API_KEY` is correctly set in the `.envrc` file
2. **Direnv not loading**: Run `direnv allow` in the project directory
3. **Python dependencies**: Make sure to run `poetry install` to install all required packages
4. **File paths**: Ensure input files exist at the specified paths

### Project Structure
