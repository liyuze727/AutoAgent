# AutoAgent Project

This is a Python project using LangChain Agents for automated data analysis, feature engineering, and clustering.
1.  **Description Agent:** Performs Exploratory Data Analysis (EDA), generates summaries/visualizations, and uses an LLM (Google Gemini) for a data context summary. Allows human feedback for refinement.
2.  **Feature Engineering Agent:** Based on the EDA context, proposes new features using the LLM and attempts to engineer them using pandas `eval`.
3.  **Clustering Agent:** Takes the (potentially feature-engineered and preprocessed) data and applies multiple clustering algorithms (KMeans, GMM, Agglomerative, DBSCAN), evaluates them using Silhouette Score, selects the best, and generates a PCA visualization of the clusters.

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
2.  In the **same terminal**, run the main script:
    ```bash
    python3 main.py
    ```
3.  **Description Agent:** Runs first, performs EDA, calls LLM. Pauses for approval (`Do you approve... (yes/no):`).
    * `yes`/`y`: Proceeds to Feature Engineering.
    * `no`/`n`: Prompts for feedback, revises, asks again.
4.  **Feature Engineering Agent:** Runs after description approval. Proposes and attempts to create features. Prints a report.
5.  **Clustering Agent:** Runs after feature engineering (using the potentially modified and preprocessed data).
    * Runs KMeans, GMM, Agglomerative, DBSCAN.
    * Evaluates results using Silhouette Score (penalizes high DBSCAN noise).
    * Prints an evaluation summary.
    * Selects the best method.
    * Generates and attempts to display a PCA visualization of the best clustering result (display requires a compatible environment like Jupyter or IPython).and **saves it as `cluster_visualization.png`** in the project directory. It also attempts to display the image inline if the environment supports it (e.g., Jupyter).

## File Structure

* `main.py`: Main script, runs data loading, preprocessing, and all agents sequentially.
* `agents.py`: Defines `DescriptionAgent`, `FeatureEngAgent`, and `ClusteringAgent` classes and their graph logic.
* `utils.py`: Contains EDA, context building, and LLM query helper functions.
* `requirements.txt`: Required Python libraries.
* `README.md`: (This file) Project description and setup guide.

