# AutoAgent Data Analysis and Feature Engineering Project

This is a Python project using LangChain Agents for automated data analysis and preliminary feature engineering suggestions. It first performs Exploratory Data Analysis (EDA) on the input dataset, generating text summaries and visualizations. Then, it utilizes a Large Language Model (LLM, Google Gemini in this case) to synthesize a data context summary. Subsequently, based on this summary and additional user input, it can further suggest or execute feature engineering steps (the current feature engineering part is a simplified example).

## Prerequisites

* **Python 3:** Ensure you have Python 3 installed on your system (version 3.9 or higher recommended). If not, download and install it from [python.org](https://www.python.org/). Installing Python usually installs `pip3` automatically.
* **pip3:** Python's package manager. If your system distinguishes between `pip` (Python 2) and `pip3` (Python 3), please use `pip3`. You can check by running `pip3 --version` in your terminal.

## Setup

1.  **Clone or Download Project:** Place the `agents.py`, `main.py`, `utils.py`, and `requirements.txt` files in the same project folder (e.g., `AutoAgent`).
2.  **Install Dependencies:** Open a terminal, navigate to the project folder, and run:
    ```bash
    pip3 install -r requirements.txt
    ```
3.  **Get Google API Key:**
    * Visit [Google AI Studio](https://aistudio.google.com/).
    * Log in with your Google account.
    * Click "Get API key" and follow the instructions to create a new API key (you might need to associate it with a Google Cloud project).
    * Copy the generated API key. **Please keep it safe and do not share it publicly.**
4.  **Set Environment Variable:** **Before running the script**, you need to set the `GOOGLE_API_KEY` environment variable in your terminal. **Do not hardcode the key directly into the code!**
    * **Linux / macOS:**
        ```bash
        export GOOGLE_API_KEY='your_api_key_here'
        ```
    * **Windows (Command Prompt):**
        ```bash
        set GOOGLE_API_KEY=your_api_key_here
        ```
    * **Windows (PowerShell):**
        ```bash
        $env:GOOGLE_API_KEY = 'your_api_key_here'
        ```
    * **Note:** Replace `'your_api_key_here'` with your actual API key. This setting is typically only valid for the current terminal session.

## Running the Project

1.  Ensure you have completed all the setup steps above (especially setting the API key environment variable).
2.  In the **same terminal** where you set the environment variable, run the main script:
    ```bash
    python3 main.py
    ```
3.  The program will load the data and run the `DescriptionAgent`. After the `DescriptionAgent`'s LLM outputs, it will pause in the terminal and ask for your approval (`Do you approve the LLM's output? (yes/no):`).
    * Enter `yes` (or `y`) to approve and continue (if the `FeatureEngAgent` is configured to run next).
    * Enter `no` (or `n`) to disapprove. The program will then prompt you for improvement suggestions (`What aspect should be improved...`), and call the LLM again for revision.
4.  (If `run_fe_agent` is set to `True` in `main.py`) The program will proceed to run the feature engineering steps (currently a simplified example).

## File Structure

* `main.py`: Main execution script.
* `agents.py`: Defines the `DescriptionAgent` and `FeatureEngAgent` classes.
* `utils.py`: Contains functions for data processing, EDA, and LLM helper calls.
* `requirements.txt`: List of required Python libraries for the project.
* `README.md`: (This file) Project description and setup guide.

