# main.py
# Main script to run the data analysis and feature engineering agents.

import pandas as pd
import uuid
from IPython.display import display, Image, Markdown # For potential display in non-notebook envs
import base64 # For displaying images if needed outside notebook

# Import agent classes and utility functions
try:
    from agents import DescriptionAgent, FeatureEngAgent # Assuming agents.py is accessible
    from utils import build_data_context_from_df, ask_about_context # Assuming utils.py is accessible
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure agents.py and utils.py are in the same directory or Python path.")
    exit()

# Import LLM and data loading components
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.datasets import fetch_openml
# --- Configuration ---
# It's highly recommended to use environment variables or a secrets manager
# for API keys in production code. Hardcoding is insecure.
import os
# from google.colab import userdata # This only works in Colab
# GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY') # Colab method
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY') # Standard environment variable method

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set the GOOGLE_API_KEY environment variable before running.")
    # You might want to exit() here in a real application
    # For demonstration, we'll let it proceed but LLM calls will fail.
    # exit()


# --- LLM Initialization ---
try:
    llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.2,
                convert_system_message_to_human=True
            )
    print("LLM Initialized Successfully.")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    llm = None # Set LLM to None if initialization fails


# --- Data Loading ---
print("\nLoading dataset...")
df_input_data = None
try:
    # Using wine-quality-red as in the notebook example
    df_input_data, _ = fetch_openml('wine-quality-red', version=1, return_X_y=True, as_frame=True)
    print(f"Dataset 'wine-quality-red' loaded successfully. Shape: {df_input_data.shape}")
except Exception as e:
    print(f"Failed to load OpenML dataset: {e}")
    # Add alternative data loading or exit if data is crucial
    # df_input_data = pd.read_csv("your_local_file.csv") # Example local load


# --- Main Execution Logic ---
if df_input_data is not None and llm is not None:
    # === Run Description Agent ===
    print("\n--- Running Description Agent ---")
    description_agent = DescriptionAgent(llm)
    # Start a new thread for the description task
    desc_thread_id = uuid.uuid4().hex
    desc_final_state, desc_last_msg = description_agent.run(df_input_data, thread_id=desc_thread_id)

    if desc_final_state and desc_last_msg:
        print("\n--- Description Agent Final Output ---")
        print(desc_last_msg.content) # Display the final approved description

        # Store results needed for the next agent
        initial_grams = desc_final_state.get("grams")
        initial_messages = desc_final_state.get("messages")

        # === Optionally Run Feature Engineering Agent ===
        run_fe_agent = True # Set to False if you only want the description

        if run_fe_agent:
            print("\n--- Running Feature Engineering Agent ---")
            # Provide some initial context/prompt if desired
            user_fe_context = "Focus on features related to acidity and alcohol interaction."
            feature_eng_agent = FeatureEngAgent(llm)

            # Pass the final state from the description agent if available
            fe_final_state = feature_eng_agent.eng(
                df=df_input_data, # Pass original or potentially modified df if needed
                description=user_fe_context,
                initial_grams=initial_grams,
                initial_messages=initial_messages
            )

            if fe_final_state:
                print("\n--- Feature Engineering Agent Final State ---")
                # Print the messages accumulated during the FE process
                fe_messages = fe_final_state.get("messages", [])
                print("Messages:")
                for i, msg in enumerate(fe_messages):
                    print(f"  {i+1}: {msg.content[:150]}...") # Print excerpt
                # You might want to inspect the final DataFrame:
                # final_df = fe_final_state.get("dataset")
                # if final_df is not None:
                #     print("\nFinal DataFrame Info:")
                #     final_df.info()

            else:
                print("\nFeature Engineering Agent did not complete successfully.")

    else:
        print("\nDescription Agent did not complete successfully or returned no message.")

elif df_input_data is None:
    print("\nExecution stopped: Data loading failed.")
elif llm is None:
     print("\nExecution stopped: LLM initialization failed.")

print("\nScript finished.")

