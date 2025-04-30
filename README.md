# AutoAgent Project

This is a Python project using LangChain Agents for automated data analysis and feature engineering. It first performs Exploratory Data Analysis (EDA) on the input dataset, generating text summaries and visualizations, and synthesizes a data context summary using a Large Language Model (LLM - Google Gemini). Subsequently, a second agent attempts to propose and engineer new features based on the EDA context and LLM suggestions, modifying the dataset.

## Prerequisites

* **Python 3:** Ensure you have Python 3 installed (version 3.9+ recommended). Comes with `pip3`. Download from [python.org](https://www.python.org/).
* **pip3:** Python's package manager. Use `pip3 --version` to check.

## Setup

1.  **Clone/Download:** Place `agents.py`, `main.py`, `utils.py`, `requirements.txt` in one project folder (e.g., `AutoAgent`).
2.  **Install Dependencies:** In the project folder terminal, run:
    ```bash
    pip3 install -r requirements.txt
    ```
3.  **Get Google API Key:**
    * Visit [Google AI Studio](https://aistudio.google.com/).
    * Log in, click "Get API key", create/select a project, and copy the key. **Keep it secret.**
4.  **Set Environment Variable:** Before running, set `GOOGLE_API_KEY` in your terminal. **Do not hardcode the key!**
    * **Linux / macOS:**
        ```bash
        export GOOGLE_API_KEY='your_api_key_here'
        ```
    * **Windows (CMD):**
        ```bash
        set GOOGLE_API_KEY=your_api_key_here
        ```
    * **Windows (PowerShell):**
        ```bash
        $env:GOOGLE_API_KEY = 'your_api_key_here'
        ```
    * Replace `'your_api_key_here'` with your actual key. This is session-specific.

## Running the Project

1.  Ensure setup is complete (API key variable set).
2.  In the **same terminal**, run:
    ```bash
    python3 main.py
    ```
3.  The `DescriptionAgent` runs first. It will pause and ask for approval (`Do you approve the LLM's output? (yes/no):`).
    * `yes` (or `y`): Approves and proceeds to the Feature Engineering agent.
    * `no` (or `n`): Prompts for feedback (`What aspect should be improved...`), revises the description, and asks for approval again.
4.  After description approval, the `FeatureEngAgent` runs automatically. It will:
    * Attempt to propose features using the LLM based on the context.
    * Attempt to engineer these features by evaluating formulas.
    * Print a report of successful/failed feature engineering attempts.
    * Display information about the final DataFrame (potentially with new columns).

## File Structure

* `main.py`: Main execution script, runs both agents.
* `agents.py`: Defines `DescriptionAgent` and `FeatureEngAgent` classes and their graph logic.
* `utils.py`: Contains EDA, context building, and LLM query helper functions.
* `requirements.txt`: Required Python libraries.
* `README.md`: (This file) Project description and setup guide.

