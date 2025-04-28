# utils.py
# Contains helper functions for EDA, context building, and querying.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings
import json
from typing import Dict, List, Tuple, Optional, Any

# Import necessary sklearn components
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Import Langchain components (ensure these are installed)
# Note: LLM client initialization is moved to main.py or a config file
# to avoid circular dependencies and keep utils focused.
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Suppress warnings (optional)
warnings.filterwarnings('ignore')

# ==============================================================================
# Consolidated Context Building Function
# ==============================================================================

def build_data_context_from_df(df_input: pd.DataFrame, llm_client: ChatGoogleGenerativeAI) -> Dict[str, Any]:
    """
    Performs EDA, PCA Representation & Viz, and LLM Synthesis using a provided LLM client.
    Returns all results in a dictionary.

    Args:
        df_input: The pandas DataFrame to analyze.
        llm_client: An initialized ChatGoogleGenerativeAI client.

    Returns:
        A dictionary containing all results ('status', 'error_message',
        'textual_summaries', 'standard_visualizations_b64',
        'representation_method', 'representation_visualization_b64',
        'final_llm_summary').
    """
    print("="*60)
    print("STARTING TASK 1: Advanced Context Building")
    print("="*60)

    # --- Initialize Results Dictionary ---
    results = {
        'status': 'Failure', # Default status
        'error_message': None,
        'textual_summaries': None,
        'standard_visualizations_b64': None,
        'representation_method': None,
        'representation_visualization_b64': None,
        'final_llm_summary': None
    }

    if llm_client is None:
        results['error_message'] = "LLM Client not provided."
        print("\nERROR: LLM Client not provided.")
        return results

    # --- Main Task Logic ---
    try:
        # === Task 1a: Standard EDA Logic ===
        print("--- Task 1a: Generating Textual Summary ---")
        summaries = {}
        buffer = io.StringIO()
        try:
            df_input.info(buf=buffer)
            summaries['info'] = buffer.getvalue()
        except Exception as e:
            summaries['info'] = f"Error getting info: {e}"

        try:
            # Describe might fail on mixed types if not handled carefully
            with pd.option_context('display.max_columns', 50):
                 summaries['description'] = df_input.describe(include='all').to_string()
        except Exception as e:
             summaries['description'] = f"Error getting description: {e}"

        try:
            missing_percent = (df_input.isnull().sum() / len(df_input)) * 100
            missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
            summaries['missing_values'] = missing_percent.to_string() if not missing_percent.empty else "No missing values found."
        except Exception as e:
            summaries['missing_values'] = f"Error calculating missing values: {e}"

        try:
            summaries['unique_counts'] = df_input.nunique().to_string()
        except Exception as e:
            summaries['unique_counts'] = f"Error getting unique counts: {e}"

        try: # Categorical value counts
            categorical_cols = df_input.select_dtypes(include=['object', 'category']).columns
            value_counts_summary = []
            max_cats=10
            max_cols=20
            proc_count=0
            for col in categorical_cols:
                if proc_count >= max_cols:
                    value_counts_summary.append(f"\n[Stopped value counts after {max_cols} categorical columns]")
                    break
                try:
                    nunique = df_input[col].nunique(dropna=False)
                    if 1 < nunique <= 50: # Only show counts for low/medium cardinality
                        plot_data = df_input[col].astype(str).fillna('__MISSING__') # Handle potential NaNs explicitly
                        counts = plot_data.value_counts()
                        value_counts_summary.append(f"\n'{col}' (Top {max_cats}):\n{counts.head(max_cats).to_string()}")
                        if len(counts) > max_cats:
                            value_counts_summary.append(f"... ({len(counts)-max_cats} more unique values)")
                        proc_count += 1
                    elif nunique > 50:
                         value_counts_summary.append(f"\n'{col}': Skipped (> {nunique} unique values)")
                         proc_count += 1 # Count skipped columns too
                    # else: # Skip columns with only 1 unique value (or none)
                    #    value_counts_summary.append(f"\n'{col}': Skipped (<= 1 unique value)")
                    #    proc_count += 1
                except Exception as e_inner:
                    value_counts_summary.append(f"\nError processing value counts for '{col}': {e_inner}")
                    proc_count += 1
            summaries['value_counts_categorical'] = "\n".join(value_counts_summary) if value_counts_summary else "No low/medium cardinality categorical columns found."
        except Exception as e:
            summaries['value_counts_categorical'] = f"Error processing categorical value counts: {e}"
        print("--- Textual Summary Generation Complete ---")
        results['textual_summaries'] = summaries

        print("--- Task 1a: Generating Standard Visualizations ---")
        base64_plots_std = []
        plot_counters_std = {'hist': 0, 'count': 0}
        max_plots_per_type_std = 5 # Limit plots to avoid overwhelming output
        num_plots_generated_std = 0

        def encode_plot_std(figure: plt.Figure):
            nonlocal num_plots_generated_std
            buf = io.BytesIO()
            figure.savefig(buf, format='png', bbox_inches='tight')
            plt.close(figure) # Close the plot to free memory
            buf.seek(0)
            base64_plots_std.append(base64.b64encode(buf.read()).decode('utf-8'))
            num_plots_generated_std += 1

        # Histograms for numerical columns
        numerical_cols_std = df_input.select_dtypes(include=np.number).columns
        print(f"  Generating histograms ({min(len(numerical_cols_std), max_plots_per_type_std)} max)...")
        for col in numerical_cols_std:
            if plot_counters_std['hist'] >= max_plots_per_type_std: break
            if df_input[col].isnull().all(): continue # Skip all-NaN columns
            try:
                # Check uniqueness again to avoid plotting constants
                if df_input[col].nunique(dropna=True) > 1:
                    fig_h, ax_h = plt.subplots(figsize=(8, 4))
                    sns.histplot(df_input[col].dropna(), kde=True, bins=30, ax=ax_h)
                    ax_h.set_title(f'Distribution of {col}')
                    encode_plot_std(fig_h)
                    plot_counters_std['hist'] += 1
            except Exception as e:
                print(f"    Failed to generate histogram for {col}: {e}")

        # Count plots for categorical columns (with reasonable cardinality)
        categorical_cols_std = df_input.select_dtypes(include=['object', 'category']).columns
        print(f"  Generating count plots ({min(len(categorical_cols_std), max_plots_per_type_std)} max for low/med cardinality)...")
        for col in categorical_cols_std:
             if plot_counters_std['count'] >= max_plots_per_type_std: break
             if df_input[col].isnull().all(): continue
             try:
                 nunique = df_input[col].nunique(dropna=False) # Include NaNs in uniqueness check for plotting decision
                 if 1 < nunique <= 50: # Only plot if cardinality is manageable
                     fig_c, ax_c = plt.subplots(figsize=(10, max(5, nunique*0.3))) # Adjust height based on unique values
                     plot_data = df_input[col].astype(str).fillna('__MISSING__') # Convert to string and handle NaNs
                     sns.countplot(y=plot_data, order=plot_data.value_counts().index, ax=ax_c, palette="viridis")
                     ax_c.set_title(f'Counts of {col}')
                     encode_plot_std(fig_c)
                     plot_counters_std['count'] += 1
             except Exception as e:
                 print(f"    Failed to generate count plot for {col}: {e}")

        # Correlation heatmap for numerical columns
        print("  Generating correlation heatmap...")
        if len(numerical_cols_std) > 1:
            # Handle potential all-NaN columns before imputation
            numeric_data_corr = df_input[numerical_cols_std].dropna(axis=1, how='all')
            if not numeric_data_corr.empty:
                # Impute missing values for correlation calculation
                imp_corr = SimpleImputer(strategy='median')
                try:
                    numeric_imp_corr = imp_corr.fit_transform(numeric_data_corr)
                    df_imp_corr = pd.DataFrame(numeric_imp_corr, columns=numeric_data_corr.columns)

                    # Filter out columns with zero variance after imputation
                    df_var_corr = df_imp_corr.loc[:, df_imp_corr.std() > 1e-6] # Use a small threshold

                    if df_var_corr.shape[1] > 1: # Need at least 2 columns with variance
                        corr_mat = df_var_corr.corr()
                        fig_hm, ax_hm = plt.subplots(figsize=(min(12, corr_mat.shape[1]*0.8), min(10, corr_mat.shape[1]*0.7)))
                        sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8}, ax=ax_hm)
                        ax_hm.set_title('Correlation Matrix of Numerical Features')
                        plt.xticks(rotation=45, ha='right')
                        plt.yticks(rotation=0)
                        encode_plot_std(fig_hm)
                    else:
                        print("  Skipping heatmap: Less than 2 numerical columns with variance after imputation.")
                except Exception as e:
                    print(f"    Failed to generate correlation heatmap: {e}")
            else:
                 print("  Skipping heatmap: No valid numerical data after dropping all-NaN columns.")
        else:
            print("  Skipping heatmap: Less than 2 numerical columns.")
        print(f"--- Standard Visualization Generation Complete ({num_plots_generated_std} plots) ---")
        results['standard_visualizations_b64'] = base64_plots_std

        # === Task 1b: Representation Learning (PCA) & Visualization Logic ===
        print("--- Task 1b: Preprocessing for Standard Representation ---")
        df_1b_processed = None
        numerical_cols_1b = df_input.select_dtypes(include=np.number).columns
        categorical_cols_1b = df_input.select_dtypes(exclude=np.number).columns # Select non-numeric as categorical

        if not numerical_cols_1b.empty or not categorical_cols_1b.empty:
            transformers_1b = []
            if not numerical_cols_1b.empty:
                transformers_1b.append(('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')), # Impute numerical
                    ('scaler', StandardScaler())]), numerical_cols_1b))
                print(f"  Preprocessing numerical features ({len(numerical_cols_1b)}).")
            if not categorical_cols_1b.empty:
                transformers_1b.append(('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')), # Impute categorical
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_cols_1b))
                print(f"  Preprocessing categorical features ({len(categorical_cols_1b)}).")

            preprocessor_1b = ColumnTransformer(transformers=transformers_1b, remainder='drop')

            try:
                df_transformed_1b = preprocessor_1b.fit_transform(df_input)
                feature_names_out_1b = preprocessor_1b.get_feature_names_out()
                df_1b_processed = pd.DataFrame(df_transformed_1b, columns=feature_names_out_1b, index=df_input.index)
                print(f"  Preprocessing complete. Processed shape: {df_1b_processed.shape}")

                # Check for NaNs/Infs *after* transformation (though imputation should prevent this)
                if df_1b_processed.isnull().any().any() or np.isinf(df_1b_processed.values).any():
                    print("  Warning: NaNs or Infs detected after preprocessing. Applying secondary imputation.")
                    # Apply imputation again just in case (e.g., if OHE created issues not caught)
                    final_imputer = SimpleImputer(strategy='median')
                    df_1b_processed = pd.DataFrame(final_imputer.fit_transform(df_1b_processed),
                                                   columns=df_1b_processed.columns,
                                                   index=df_1b_processed.index)
            except ValueError as ve:
                 # More specific error for empty data after selection
                 if "No feature was selected" in str(ve):
                     print(f"  Error during preprocessing: No features remained after selection/transformation. Skipping representation.")
                 else:
                     print(f"  Error during preprocessing: {ve}")
                 df_1b_processed = None # Ensure it's None on error
            except Exception as e:
                print(f"  Error during preprocessing: {e}")
                df_1b_processed = None # Ensure it's None on error
        else:
            print("  Error: No usable columns found in the input DataFrame for preprocessing.")


        print("--- Task 1b: Generating Representation (PCA) ---")
        representation_df_1b = None
        method_used_1b = "PCA (Skipped)"
        if df_1b_processed is not None and not df_1b_processed.empty:
            # Double-check for NaNs/Infs before PCA
            if not (df_1b_processed.isnull().values.any() or np.isinf(df_1b_processed.values).any()):
                try:
                    variance_threshold=0.90 # Target variance explained
                    n_components_1b = None
                    current_n_features_1b = df_1b_processed.shape[1]
                    current_n_samples_1b = df_1b_processed.shape[0]

                    # PCA n_components must be <= min(n_samples, n_features)
                    max_possible_comps_1b = min(current_n_samples_1b, current_n_features_1b)

                    if max_possible_comps_1b >= 1:
                        # Determine number of components needed for the threshold
                        pca_find_k_1b = PCA(n_components=max_possible_comps_1b, random_state=42)
                        pca_find_k_1b.fit(df_1b_processed)
                        cumulative_variance_1b = np.cumsum(pca_find_k_1b.explained_variance_ratio_)
                        # Find the first index where cumulative variance exceeds threshold
                        n_components_calc_1b = np.argmax(cumulative_variance_1b >= variance_threshold) + 1
                        n_components_1b = max(1, int(n_components_calc_1b)) # Ensure at least 1 component

                        print(f"  Selected n_components = {n_components_1b} for >= {variance_threshold*100:.0f}% variance.")

                        if n_components_1b >= 1:
                            pca_1b = PCA(n_components=n_components_1b, random_state=42)
                            pca_result_1b = pca_1b.fit_transform(df_1b_processed)
                            explained_variance_1b = pca_1b.explained_variance_ratio_.sum()
                            print(f"  PCA complete. Result shape: {pca_result_1b.shape}, Explained Variance: {explained_variance_1b:.2f}")

                            representation_df_1b = pd.DataFrame(pca_result_1b, index=df_1b_processed.index,
                                                                columns=[f'PC_{i+1}' for i in range(n_components_1b)])
                            method_used_1b = f"PCA ({n_components_1b} components, {explained_variance_1b*100:.0f}% variance)"
                        else:
                             print("  Skipping PCA: Calculated number of components is less than 1.")
                    else:
                        print(f"  Skipping PCA: Cannot perform PCA with {max_possible_comps_1b} possible components (needs >= 1).")
                except Exception as e:
                    print(f"  Error during PCA execution: {e}")
                    method_used_1b = "PCA (Error)"
            else:
                 print("  Skipping PCA: Input data contains NaNs or Infs after preprocessing.")
        elif df_1b_processed is None:
             print("  Skipping PCA: Preprocessing failed.")
        else: # df_1b_processed is empty
             print("  Skipping PCA: Preprocessed data is empty.")
        results['representation_method'] = method_used_1b

        print("--- Task 1b: Visualizing Representation (t-SNE on PCA results) ---")
        rep_viz_b64_1b = None
        # Visualize only if PCA was successful and produced >= 2 components
        if representation_df_1b is not None and not representation_df_1b.empty and representation_df_1b.shape[1] >= 2:
             # Check for NaNs/Infs in PCA results before t-SNE
            if representation_df_1b.isnull().values.any() or np.isinf(representation_df_1b.values).any():
                print("  Warning: NaNs or Infs detected in PCA results. Applying imputation before t-SNE.")
                imputer_tsne = SimpleImputer(strategy='mean')
                representation_df_1b = pd.DataFrame(imputer_tsne.fit_transform(representation_df_1b),
                                                    index=representation_df_1b.index,
                                                    columns=representation_df_1b.columns)
            try:
                print(f"  Running t-SNE on PCA results (shape: {representation_df_1b.shape})...")
                n_samples_viz_1b = representation_df_1b.shape[0]

                # t-SNE requires n_samples > perplexity + 1
                # Adjust perplexity based on sample size
                perplexity_value_1b = min(30.0, max(5.0, float(n_samples_viz_1b - 1)))
                if n_samples_viz_1b <= perplexity_value_1b + 1 : # Adjust if needed
                    perplexity_value_1b = max(1.0, float(n_samples_viz_1b - 2)) # Ensure perplexity < n_samples - 1
                    print(f"  Adjusted t-SNE perplexity to {perplexity_value_1b:.1f} due to small sample size.")

                if n_samples_viz_1b >= 3 and perplexity_value_1b >= 1: # Need at least 3 samples for t-SNE usually
                     # Use auto learning rate and PCA initialization which are often more robust
                     reducer_1b = TSNE(n_components=2, perplexity=perplexity_value_1b,
                                       random_state=42, n_iter=300, init='pca', learning_rate='auto')
                     reduced_data_1b = reducer_1b.fit_transform(representation_df_1b)

                     # Generate the plot
                     fig_viz_1b, ax_viz_1b = plt.subplots(figsize=(8, 6))
                     ax_viz_1b.scatter(reduced_data_1b[:, 0], reduced_data_1b[:, 1], s=10, alpha=0.7, cmap='viridis') # Removed c= argument if no color info
                     ax_viz_1b.set_title(f't-SNE Visualization of {method_used_1b}')
                     ax_viz_1b.set_xlabel('t-SNE Component 1')
                     ax_viz_1b.set_ylabel('t-SNE Component 2')
                     ax_viz_1b.grid(True, linestyle='--', alpha=0.5)

                     # Encode the plot
                     buf_1b = io.BytesIO()
                     fig_viz_1b.savefig(buf_1b, format='png', bbox_inches='tight')
                     plt.close(fig_viz_1b)
                     buf_1b.seek(0)
                     rep_viz_b64_1b = base64.b64encode(buf_1b.read()).decode('utf-8')
                     print("  t-SNE visualization generated.")
                else:
                     print(f"  Skipping t-SNE: Insufficient samples ({n_samples_viz_1b}) or invalid perplexity ({perplexity_value_1b:.1f}).")

            except Exception as e:
                print(f"  Error during t-SNE visualization: {e}")
        elif representation_df_1b is not None and representation_df_1b.shape[1] < 2:
             print("  Skipping t-SNE visualization: PCA resulted in less than 2 components.")
        else: # PCA failed or produced empty results
            print("  Skipping t-SNE visualization: Invalid or missing PCA representation data.")
        results['representation_visualization_b64'] = rep_viz_b64_1b

        # === Task 1c: LLM Synthesis Logic ===
        print("--- Task 1c: Synthesizing Context with LLM ---")
        final_summary_1c = None
        prompt_content_1c = []

        # --- Build the Prompt ---
        # Start with the textual instruction for the LLM
        prompt_text = f"""You are an expert data analyst building context for an ML task (likely clustering or prediction later). Synthesize the following information into a comprehensive 'Data Context Summary'. The goal is deep data understanding.

Dataset Columns: {list(df_input.columns)}
Dataset Shape: {df_input.shape}

--- Standard EDA Textual Summary ---
{json.dumps(results.get('textual_summaries', {'info': 'Not Available'}), indent=2, ensure_ascii=False)}

--- Advanced Representation Analysis ---
Method Used: {results.get('representation_method', 'Not Available')}
Visualization: A t-SNE plot visualizing the learned representation is provided as an image below (if generated). Analyze this plot for potential structure, groupings, density variations, or lack thereof.

--- Analysis Task ---
Generate a comprehensive summary focusing on data understanding. Integrate ALL available information (text summaries, standard plots provided as images, representation plot provided as an image). Cover:
1.  Likely data domain and purpose based on column names and data patterns.
2.  Key characteristics: data types, presence of missing values, overall size.
3.  Distributions, correlations, and patterns observed from the standard EDA (referencing summaries and implicitly the standard visualizations). Mention skewness, potential outliers, and key relationships.
4.  Insights derived from the visualization of the learned representation (e.g., "The t-SNE plot of the {results.get('representation_method', 'representation')} suggests X potential groupings based on visual density..." or "The representation appears scattered without clear structure, indicating..."). If no plot was generated, state that.
5.  Potential data quality issues or areas needing further investigation (missing data patterns, outliers suggested by stats/plots, high cardinality categoricals).
6.  Overall assessment of data complexity and potential underlying structure based on all available information.
7.  Comment briefly on the potential suitability of advanced models like TabPFN if data size/characteristics seem appropriate (e.g., small dataset size < ~2k samples, mostly numerical/low-cardinality categorical).

Provide ONLY the final summary narrative, structured clearly. Focus on describing the data's meaning and characteristics for context building. Do not repeat the input summaries verbatim; synthesize them.
"""
        prompt_content_1c.append({"type": "text", "text": prompt_text})

        # Append standard visualizations (if they exist)
        valid_std_plots = 0
        std_viz_list = results.get('standard_visualizations_b64', [])
        if std_viz_list:
            prompt_content_1c.append({"type": "text", "text": "\n--- Standard Visualizations (Images) ---"})
            for i, img_b64 in enumerate(std_viz_list):
                if isinstance(img_b64, str) and len(img_b64) > 100: # Basic check for valid base64
                     try:
                         # Validate decoding (optional but good practice)
                         base64.b64decode(img_b64)
                         prompt_content_1c.append({
                             "type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                         })
                         valid_std_plots += 1
                     except Exception as img_err:
                         print(f"    Skipping invalid base64 string for standard plot {i+1}: {img_err}")
                else:
                     print(f"    Skipping invalid or short data for standard plot {i+1}")
            if valid_std_plots == 0:
                prompt_content_1c.append({"type": "text", "text": "[No valid standard plots generated or provided]"})
        else:
             prompt_content_1c.append({"type": "text", "text": "\n[No Standard Plots Provided]"})

        # Append representation visualization (if it exists)
        prompt_content_1c.append({"type": "text", "text": f"\n--- Representation Visualization ({results.get('representation_method','N/A')}) (Image) ---"})
        rep_viz_b64_1c = results.get('representation_visualization_b64')
        if rep_viz_b64_1c and isinstance(rep_viz_b64_1c, str) and len(rep_viz_b64_1c) > 100:
            try:
                base64.b64decode(rep_viz_b64_1c) # Validate
                prompt_content_1c.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{rep_viz_b64_1c}"}
                })
            except Exception as img_err:
                 print(f"    Skipping invalid base64 string for representation plot: {img_err}")
                 prompt_content_1c.append({"type": "text", "text": "[Representation plot data invalid]"})
        else:
             prompt_content_1c.append({"type": "text", "text": "[Representation plot not available or invalid]"})

        # --- Invoke LLM ---
        try:
            print(f"  Invoking LLM ({llm_client.model}) for final context synthesis...")
            # Use the passed llm_client
            message = llm_client.invoke([HumanMessage(content=prompt_content_1c)])
            final_summary_1c = message.content
            print("  Final synthesis complete.")
        except Exception as e:
            print(f"  Error during final LLM synthesis: {e}")
            final_summary_1c = f"Error during final synthesis: {e}"
            results['error_message'] = final_summary_1c # Store LLM specific error

        results['final_llm_summary'] = final_summary_1c

        # Final status update based on LLM result
        if final_summary_1c and not final_summary_1c.startswith("Error"):
            results['status'] = 'Success'
        else:
            results['status'] = 'Failure' # Keep status as Failure if LLM errored or no summary

    except Exception as e: # Catch errors in the main try block for tasks 1a/1b/1c
        import traceback
        print(f"An unexpected error occurred during Task 1 execution: {e}")
        print(traceback.format_exc()) # Print detailed traceback for debugging
        results['error_message'] = f"Unexpected error in main execution: {e}"
        results['status'] = 'Failure' # Ensure status is Failure

    print("\n" + "="*60)
    print(f"TASK 1 Complete. Status: {results['status']}")
    print("="*60)

    return results


# ==============================================================================
# Function to Ask About / Refine Data Context
# ==============================================================================

def ask_about_context(
    task1_results: Dict[str, Any],
    user_message: str, # This message now drives refinement/Q&A
    llm_client: ChatGoogleGenerativeAI # Pass the initialized client
    ) -> Optional[str]:
    """
    Uses the LLM to answer a user's question OR refine the data context summary,
    based on the user's message and previously generated Task 1 results.
    Does NOT use images for refinement/Q&A to keep it focused.

    Args:
        task1_results: The dictionary output from build_data_context_from_df.
        user_message: The user's question, feedback, or refinement request.
        llm_client: The initialized ChatGoogleGenerativeAI client.

    Returns:
        The LLM's response string (either an answer or a refined summary),
        or an error message string. Returns None if inputs are invalid.
    """
    print("\n--- Task 2: Asking About / Refining Context ---")

    if not llm_client:
        print("  Error: LLM Client not available for context query.")
        return "Error: LLM Client not available."

    # Check if task1_results are valid and successful before proceeding
    if not isinstance(task1_results, dict) or task1_results.get('status') != 'Success':
        error_msg = task1_results.get('error_message', 'Task 1 results dictionary is invalid or task failed.') if isinstance(task1_results, dict) else "Task 1 results are missing or invalid."
        print(f"  Error: Cannot query/refine context. Reason: {error_msg}")
        return f"Error: Cannot query/refine context. Reason: {error_msg}"

    # Extract necessary context information from Task 1 results
    original_summary = task1_results.get('final_llm_summary', 'No previous context summary available.')
    text_summaries = task1_results.get('textual_summaries', {})
    representation_method = task1_results.get('representation_method', 'No representation method available.')

    # Provide key stats for grounding, keep it reasonably concise
    # Safely access nested dictionary keys
    desc_summary = text_summaries.get('description', 'N/A')
    missing_summary = text_summaries.get('missing_values', 'N/A')
    context_details = f"Original EDA Description Stats (excerpt):\n{desc_summary[:1000]}...\n\nMissing Values Info:\n{missing_summary}"

    # Construct the dynamic prompt for Q&A or Refinement (Text Only)
    prompt = f"""You are an expert data analyst assistant. You previously analyzed a dataset and generated an initial context summary.

    --- Key Original EDA Details (for reference) ---
    {context_details}

    --- Previous Context Summary (for reference) ---
    {original_summary}
    --- End of Previous Summary ---

    --- Advanced Representation Method Used (for reference) ---
    {representation_method}
    (Note: Visualizations are not provided for this query/refinement step)

    --- User's Latest Input ---
    {user_message}
    --- End of User Input ---

    **Your Task:**
    Carefully analyze the "User's Latest Input".
    1.  If the input is feedback or asks for a refined perspective (e.g., "focus more on business context", "explain the correlations in simpler terms", "rewrite the summary emphasizing potential issues"), **generate an updated context summary** incorporating that request. Base the refinement *only* on the original EDA details and previous summary provided above.
    2.  If the input is a specific question about the data (e.g., "Which feature had the most missing values?", "What does the PCA representation imply according to the summary?"), **answer that question directly**, using the original EDA details and previous summary as your knowledge base.
    3.  Do not add external knowledge or analyze images (as they are not provided here). Ground your response solely in the textual information provided.
    4.  If the user asks about visualizations, state that they were generated previously but are not available for this specific query.

    **Response (Refined Summary or Answer):**
    """

    try:
        print(f"  Sending request to LLM based on user input: '{user_message[:100]}...'")
        # Invoke the LLM with the text-based prompt
        response_message = llm_client.invoke([HumanMessage(content=prompt)])
        response_text = response_message.content
        print("  Response received from LLM.")
        return response_text
    except Exception as e:
        print(f"  Error invoking LLM for query/refinement: {e}")
        return f"Error querying/refining context: {e}"

