# main.py
# Main script to run the data analysis and feature engineering agents.
# V7: Always save clustering visualization to a file after attempting display.

import pandas as pd
import numpy as np
import uuid
from IPython.display import display, Image, Markdown # For potential display
import base64 # For displaying images if needed
import os
import json # Keep import just in case

# Import agent classes and utility functions
try:
    from agents import DescriptionAgent, FeatureEngAgent, ClusteringAgent
    from utils import build_data_context_from_df, ask_about_context
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure agents.py and utils.py are in the same directory or Python path.")
    exit()

# Import LLM, data loading, and preprocessing components
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from langchain_core.messages import AIMessage, HumanMessage

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    llm = None
else:
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
        llm = None

# --- Data Loading ---
print("\nLoading dataset...")
df_input_data = None
try:
    df_input_data, _ = fetch_openml('wine-quality-red', version=1, return_X_y=True, as_frame=True)
    print(f"Dataset 'wine-quality-red' loaded successfully. Shape: {df_input_data.shape}")
except Exception as e:
    print(f"Failed to load OpenML dataset: {e}")

# --- Data Preprocessing ---
X_processed = None
processed_feature_names = None
preprocessor = None

if df_input_data is not None:
    print("\nPreprocessing data for agents...")
    try:
        numerical_cols = df_input_data.select_dtypes(include=np.number).columns
        categorical_cols = df_input_data.select_dtypes(exclude=np.number).columns
        print(f"Numerical cols: {list(numerical_cols)}, Categorical cols: {list(categorical_cols)}")

        transformers = []
        if len(numerical_cols) > 0:
            numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
            transformers.append(('num', numeric_transformer, numerical_cols))
        if len(categorical_cols) > 0:
            categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
            transformers.append(('cat', categorical_transformer, categorical_cols))

        if not transformers:
            print("  Error: No columns to preprocess.")
        else:
            preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
            X_processed = preprocessor.fit_transform(df_input_data)
            try: processed_feature_names = list(preprocessor.get_feature_names_out())
            except Exception: processed_feature_names = None
            print(f"Preprocessing successful. Processed data shape: {X_processed.shape}")
            if np.isnan(X_processed).any() or np.isinf(X_processed).any():
                 print("  Warning: NaNs/Infs detected after preprocessing. Applying secondary imputation.")
                 final_imputer = SimpleImputer(strategy='median'); X_processed = final_imputer.fit_transform(X_processed)
                 if np.isnan(X_processed).any() or np.isinf(X_processed).any(): print("  Error: NaNs/Infs persist."); X_processed = None
    except Exception as e:
        print(f"  Error during preprocessing: {e}"); X_processed = None

# --- Main Execution Logic ---
desc_final_state = None
fe_final_state = None
clustering_final_state = None

if df_input_data is not None and llm is not None:
    # === Run Description Agent ===
    print("\n--- Running Description Agent ---")
    description_agent = DescriptionAgent(llm)
    desc_thread_id = uuid.uuid4().hex
    desc_final_state, desc_last_msg = description_agent.run(df_input_data.copy(), thread_id=desc_thread_id)

    # Proceed only if description agent was successful
    if desc_final_state and desc_last_msg and isinstance(desc_last_msg, AIMessage) and not desc_final_state.get("error_message"):
        print("\n--- Description Agent Final Output ---")
        print(desc_last_msg.content)

        # === Run Feature Engineering Agent ===
        print("\n--- Running Feature Engineering Agent ---")
        feature_eng_agent = FeatureEngAgent(llm)
        fe_final_state = feature_eng_agent.eng(df=df_input_data.copy(), description_state=desc_final_state)

        # Proceed only if feature engineering agent was successful
        if fe_final_state and not fe_final_state.get("error_message"):
            print("\n--- Feature Engineering Agent Final State ---")
            print("\nFeature Engineering Report:")
            print(fe_final_state.get("feature_report", "No report generated."))

            final_engineered_df = fe_final_state.get("dataset")
            if isinstance(final_engineered_df, pd.DataFrame):
                print("\nEngineered DataFrame Info:")
                final_engineered_df.info()
                df_for_clustering = final_engineered_df

                if final_engineered_df.shape[1] != df_input_data.shape[1] and preprocessor:
                     print("\nRe-preprocessing data after feature engineering...")
                     try:
                         X_processed = preprocessor.fit_transform(df_for_clustering)
                         processed_feature_names = list(preprocessor.get_feature_names_out())
                         print(f"Re-preprocessing successful. Processed shape: {X_processed.shape}")
                         if np.isnan(X_processed).any() or np.isinf(X_processed).any():
                              print("  Warning: NaNs/Infs after re-preprocessing. Imputing.")
                              final_imputer = SimpleImputer(strategy='median'); X_processed = final_imputer.fit_transform(X_processed)
                              if np.isnan(X_processed).any() or np.isinf(X_processed).any(): print("  Error: NaNs/Infs persist."); X_processed = None
                     except Exception as e: print(f"  Re-preprocessing error: {e}"); X_processed = None
                elif X_processed is None:
                     print("Error: Initial preprocessing failed, cannot proceed.")
                else:
                     print("\nUsing initially preprocessed data for clustering.")

            else:
                 print("\nEngineered dataset is not a DataFrame. Cannot proceed.")
                 X_processed = None

        else:
            print("\nFeature Engineering Agent did not complete successfully or returned an error.")
            print(f"  Error: {fe_final_state.get('error_message', 'Unknown FE error') if fe_final_state else 'FE Agent returned None'}")
            if X_processed is None: print("Initial preprocessing also failed. Cannot proceed.")
            else: print("Proceeding to clustering with initially preprocessed data.")

        # === Run Clustering Agent ===
        if X_processed is not None:
            print("\n--- Running Clustering Agent ---")
            clustering_agent = ClusteringAgent()
            clustering_final_state = clustering_agent.cluster(
                preprocessed_data=X_processed,
                feature_names=processed_feature_names
            )

            if clustering_final_state and not clustering_final_state.get("error_message"):
                print("\n--- Clustering Agent Final State ---")
                print("\nEvaluation Summary:")
                eval_summary = clustering_final_state.get('evaluation_summary', {})
                if eval_summary:
                    for method, details in eval_summary.items():
                         valid_str = "(Selected as Best)" if method == clustering_final_state.get('best_method') else \
                                     ("(Considered Valid)" if details.get('is_valid_candidate') else "(Invalid/Penalized)")
                         print(f"  - {method}: Score={details.get('score', -1.1):.4f}, "
                               f"Noise Ratio={details.get('noise_ratio', 0):.2%} {valid_str}")
                         print(f"    Params: {details.get('params', {})}")
                else:
                    print("No evaluation summary generated.")

                print(f"\nBest Clustering Method: {clustering_final_state.get('best_method', 'N/A')}")

                img_b64 = clustering_final_state.get('visualization_base64')
                if img_b64:
                    print("\nDisplaying/Saving Best Cluster Visualization:")
                    # Try to display (works in Jupyter/IPython)
                    try:
                        display(Image(base64.b64decode(img_b64)))
                        print("  (Image displayed above if environment supports it)")
                    except NameError:
                         print("  Display function not available in this environment.")
                    except Exception as display_err:
                         print(f"  Error trying to display image: {display_err}")

                    # Always try to save to file as a fallback or primary method
                    try:
                        image_filename = "cluster_visualization.png"
                        with open(image_filename, "wb") as f:
                            f.write(base64.b64decode(img_b64))
                        print(f"  Visualization saved as '{image_filename}'")
                    except Exception as save_err:
                        print(f"  Error saving visualization to file: {save_err}")
                else:
                    print("No visualization generated or available.")

            elif clustering_final_state:
                 print("\nClustering Agent finished with an error.")
                 print(f"  Error: {clustering_final_state.get('error_message', 'Unknown clustering error')}")
            else:
                print("\nClustering Agent did not complete successfully or returned no state.")
        else:
             print("\nSkipping Clustering Agent due to preprocessing or feature engineering failure.")

    elif desc_final_state and desc_last_msg:
         print("\nDescription Agent finished, but the last message wasn't from the LLM or an error occurred.")
         print(f"  Error: {desc_final_state.get('error_message', 'Unknown description error')}")
    else:
        print("\nDescription Agent did not complete successfully or returned no message.")

elif df_input_data is None:
    print("\nExecution stopped: Data loading failed.")
elif llm is None:
     print("\nExecution stopped: LLM initialization failed.")

print("\nScript finished.")

