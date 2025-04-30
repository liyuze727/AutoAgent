# agents.py
# Defines the AgentState and the Agent classes (DescriptionAgent, FeatureEngAgent, ClusteringAgent).
# V5: Fix UndefinedVariableError for 'np' in df.eval() by passing functions directly.

import operator
import uuid
import pandas as pd
import numpy as np
import pickle # For MemorySaver serialization
import traceback
import ast
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from typing import TypedDict, Annotated, Any, List, Dict, Optional

# Langchain and Langgraph imports
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from IPython.display import display

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Import utility functions
try:
    from utils import build_data_context_from_df, ask_about_context
except ImportError:
    print("Warning: Could not import from utils.py. Ensure it's in the Python path.")
    def build_data_context_from_df(*args, **kwargs): return {'status': 'Failure', 'error_message': 'Utils not found'}
    def ask_about_context(*args, **kwargs): return "Error: Utils not found"


# ==============================================================================
# Agent State Definitions (Unchanged from v4)
# ==============================================================================

class BaseAgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    dataset: pd.DataFrame
    grams: Dict[str, Any]
    error_message: Optional[str]

class DescriptionAgentState(BaseAgentState):
    cmsg: Optional[str]

class FeatureEngAgentState(BaseAgentState):
    cmsg: Optional[str]
    proposed_features: Dict[str, str]
    feature_report: str

class ClusteringState(TypedDict):
    input_data_for_clustering: Optional[np.ndarray]
    feature_names: Optional[List[str]]
    clustering_results: Dict[str, Dict[str, Any]]
    evaluation_summary: Optional[Dict[str, Any]]
    best_method: Optional[str]
    best_labels: Optional[np.ndarray]
    best_model: Any
    visualization_base64: Optional[str]
    error_message: Optional[str]


# ==============================================================================
# Description Agent Class (Unchanged from v4)
# ==============================================================================
class DescriptionAgent:
    def __init__(self, model: ChatGoogleGenerativeAI):
        if model is None: raise ValueError("LLM model must be provided.")
        self.model = model
        g = StateGraph(DescriptionAgentState)
        g.add_node("autograph", self.autograph)
        g.add_node("llm", self.call_llm)
        g.add_node("correction", self.human_correction)
        g.add_node("save_state", self.save_state)
        g.add_edge("autograph", "llm")
        g.add_edge("correction", "llm")
        g.add_edge("llm", "save_state")
        g.add_conditional_edges("save_state", self.human_check, {"end": END, "correction": "correction"})
        g.set_entry_point("autograph")
        memory = MemorySaver(serde=pickle)
        self.graph = g.compile(checkpointer=memory)

    def autograph(self, state: DescriptionAgentState) -> Dict[str, Any]:
        print("\nExecuting Autograph Node (Description Agent)...")
        df = state.get("dataset")
        if df is None or not isinstance(df, pd.DataFrame):
             print("  Error: Dataset not found or invalid.")
             return {"grams": {'status': 'Failure', 'error_message': 'Dataset missing'}, "messages": [], "dataset": None, "error_message": "Dataset missing"}
        ctx = build_data_context_from_df(df, self.model)
        return {
            "grams": ctx, "messages": [], "cmsg": None, "dataset": df,
            "error_message": ctx.get('error_message')
        }

    def call_llm(self, state: DescriptionAgentState) -> Dict[str, list]:
        print("\nExecuting LLM Node (Description Agent)...")
        grams_result = state.get("grams")
        correction_msg = state.get("cmsg")
        current_messages = state.get("messages", [])
        if not isinstance(grams_result, dict) or grams_result.get('status') != 'Success':
            print("  Skipping LLM call: Context building failed.")
            error_msg = grams_result.get('error_message', 'Context unavailable') if isinstance(grams_result, dict) else 'Context unavailable'
            return {"messages": current_messages + [SystemMessage(content=f"LLM Call Skipped: {error_msg}")]}
        if correction_msg:
            print("  LLM called with correction message.")
            prompt_for_llm = correction_msg
        else:
            print("  LLM called for initial description.")
            df = state.get("dataset")
            df_cols = list(df.columns) if df is not None else ["Unknown"]
            prompt_for_llm = (
                f"Analyze the dataset context (features: {df_cols}) provided in the 'grams' state "
                f"and tell me the story of the data.\nFocus on insights relevant for potential ML tasks.\n"
                "***Output***\ndescription: <Your synthesized data story here>"
            )
        llm_response_content = ask_about_context(
            task1_results=grams_result, user_message=prompt_for_llm, llm_client=self.model
        )
        new_messages = current_messages + [AIMessage(content=llm_response_content or "LLM did not return a response.")]
        return {"messages": new_messages}

    def save_state(self, state: DescriptionAgentState) -> DescriptionAgentState:
        print("\nExecuting Save State Node (Description Agent)...")
        return state

    def human_check(self, state: DescriptionAgentState) -> str:
        print("\nExecuting Human Check Node (Description Agent)...")
        try:
             thread_id = state['configurable']['thread_id']
             cfg = {"configurable": {"thread_id": thread_id}}
             current_state_values = self.graph.get_state(cfg).values
        except Exception:
             print("Warning: Could not retrieve latest state from checkpointer.")
             current_state_values = state
        last_message = current_state_values.get("messages", [])[-1] if current_state_values.get("messages") else None
        if last_message and isinstance(last_message, AIMessage):
            print("\n--- LLM Output ---"); print(last_message.content); print("--------------------")
        elif last_message:
             print("\n--- Last Message ---"); print(last_message.content); print("--------------------")
        else: print("\nNo message generated yet.")
        while True:
            user_input = input("Do you approve the LLM's output? (yes/no): ").strip().lower()
            if user_input in ["yes", "y"]: print("  User approved."); return "end"
            elif user_input in ["no", "n"]: print("  User disapproved."); return "correction"
            else: print("  Invalid input.")

    def human_correction(self, state: DescriptionAgentState) -> Dict[str, Any]:
        print("\nExecuting Human Correction Node (Description Agent)...")
        last_llm_msg_content = state.get("messages", [])[-1].content if state.get("messages") else "No previous message."
        feedback = input("What aspect should be improved or what should the LLM focus on now? ")
        correction_prompt = (
             f"The previous description was:\n'{last_llm_msg_content}'\n\nMy feedback is: '{feedback}'.\n\n"
             "Please generate a revised description incorporating this feedback.\n***Output***\n"
             "description: <Your revised data story here>"
        )
        return {
            "cmsg": correction_prompt,
            "messages": state.get("messages", []) + [HumanMessage(content=feedback)]
        }

    def run(self, df: pd.DataFrame, thread_id: str | None = None) -> tuple[dict | None, AnyMessage | None]:
        if not isinstance(df, pd.DataFrame) or df.empty: print("Error: Invalid DataFrame to DescAgent."); return None, None
        if thread_id is None: thread_id = str(uuid.uuid4())
        print(f"Running DescriptionAgent on thread: {thread_id}")
        cfg = {"configurable": {"thread_id": thread_id}}
        state_input: Any = {"dataset": df}
        final_state_values = None; last_message = None
        print("\n=== Running Description Agent ===")
        try:
            for event in self.graph.stream(state_input, config=cfg, stream_mode="values"):
                 final_state_values = event; print(".", end="", flush=True)
            print("\nDescription Agent graph execution finished.")
            if final_state_values:
                 last_message = final_state_values.get("messages", [])[-1] if final_state_values.get("messages") else None
            else:
                 try:
                      final_state_values = self.graph.get_state(cfg).values
                      last_message = final_state_values.get("messages", [])[-1] if final_state_values.get("messages") else None
                 except Exception as state_err: print(f"Could not retrieve state: {state_err}"); final_state_values = {"error": "Stream ended unexpectedly"}
        except Exception as e:
             print(f"\nError during DescAgent graph execution: {e}"); traceback.print_exc()
             try:
                 final_state_values = self.graph.get_state(cfg).values
                 last_message = final_state_values.get("messages", [])[-1] if final_state_values.get("messages") else None
             except Exception as state_err: print(f"Could not retrieve state after error: {state_err}"); final_state_values = {"error": str(e)}
        return final_state_values, last_message


# ==============================================================================
# Feature Engineering Agent Class
# ==============================================================================
class FeatureEngAgent:
    """
    Proposes and engineers features based on EDA context and LLM suggestions.
    Operates on FeatureEngAgentState.
    """
    def __init__(self, model: ChatGoogleGenerativeAI):
        if model is None: raise ValueError("LLM model must be provided.")
        self.model = model
        g = StateGraph(FeatureEngAgentState)
        g.add_node("prepare_fe_state", self.prepare_fe_state)
        g.add_node("select_feature", self.select_feature)
        g.add_node("eng_feature", self.eng_feature)
        g.set_entry_point("prepare_fe_state")
        g.add_edge("prepare_fe_state", "select_feature")
        g.add_edge("select_feature", "eng_feature")
        g.add_edge("eng_feature", END)
        memory = MemorySaver(serde=pickle)
        self.graph = g.compile(checkpointer=memory)

    def prepare_fe_state(self, state: FeatureEngAgentState) -> Dict[str, Any]:
        print("\nExecuting Prepare FE State Node...")
        messages = state.get("messages", []); grams = state.get("grams", {}); dataset = state.get("dataset"); cmsg = state.get("cmsg")
        description_for_fe = "No specific description provided for FE."
        if cmsg: description_for_fe = cmsg
        elif messages:
            for msg in reversed(messages):
                if isinstance(msg, (AIMessage, SystemMessage)): description_for_fe = msg.content; break
                elif isinstance(msg, HumanMessage) and msg.content.lower() not in ['y', 'n', 'yes', 'no']: description_for_fe = msg.content; break
        print(f"  Using description for FE: {description_for_fe[:100]}...")
        if 'original_columns' not in grams and isinstance(dataset, pd.DataFrame):
            grams['original_columns'] = set(dataset.columns)
        return {
            "messages": messages, "grams": grams, "dataset": dataset, "cmsg": description_for_fe,
            "proposed_features": {}, "feature_report": "", "error_message": state.get("error_message")
        }

    def select_feature(self, state: FeatureEngAgentState) -> Dict[str, Any]:
        print("\nExecuting Select Feature Node...")
        df = state.get("dataset"); grams = state.get("grams", {}); description = state.get("cmsg", "N/A"); messages = state.get("messages", [])
        if df is None: print("Error: Dataset missing."); return {"messages": messages + [AIMessage(content="FE skipped: Dataset missing.")], "proposed_features": {}, "error_message": "Dataset missing."}
        column_list = df.columns.tolist(); eda_summary = grams.get('final_llm_summary', 'N/A')
        prompt = f"""You are a data scientist specializing in feature engineering.
Based on the following context, EDA summary, and existing features, propose 3-5 new potentially useful engineered features.
Provide a unique, valid Python variable name (snake_case) and a Python expression string using pandas/numpy operations **referencing column names directly**.

**Context/Description:** {description}
**EDA Summary:** {eda_summary}
**Existing Features:** {column_list}

**Task:** Propose 3-5 new features (interactions, ratios, polynomials, transformations like log1p). Ensure expressions are valid for `df.eval()`. Use `log1p` for log transforms (available directly). Add epsilon (1e-6) for safe division. Briefly explain reasoning.

**Output Format:** ONLY a Python dictionary string {{'new_feat_name': "formula_string"}}. Example:
{{
  "acidity_ratio": "fixed_acidity / (volatile_acidity + 1e-6)", // Reason: Balance.
  "log_residual_sugar": "log1p(residual_sugar)" // Reason: Skewness.
}}"""
        proposed_features_dict = {}; llm_raw_response = "LLM Failed"; proposal_message = "FE proposal failed."
        error_msg = state.get("error_message")
        try:
            print("  Invoking LLM for feature proposals..."); llm_response = self.model.invoke([HumanMessage(content=prompt)]); llm_raw_response = llm_response.content
            print(f"  LLM Proposal Response (raw): {llm_raw_response[:200]}...")
            start_index = llm_raw_response.find('{'); end_index = llm_raw_response.rfind('}')
            if start_index != -1 and end_index != -1 and start_index < end_index:
                dict_string = llm_raw_response[start_index : end_index + 1]
                try:
                    proposed_features_dict = ast.literal_eval(dict_string)
                    if not isinstance(proposed_features_dict, dict): raise ValueError("Not a dict.")
                    if not all(isinstance(k, str) and isinstance(v, str) for k, v in proposed_features_dict.items()): raise ValueError("Keys/values not strings.")
                    proposal_message = f"LLM proposed features: {list(proposed_features_dict.keys())}"; print(f"  Parsed features: {list(proposed_features_dict.keys())}")
                except Exception as parse_err: print(f"  Parsing error: {parse_err}"); proposal_message = f"Parsing failed: {parse_err}"; error_msg = proposal_message
            else: print("  Dict structure not found."); proposal_message = "No dict found."; error_msg = proposal_message
        except Exception as e: print(f"  LLM error: {e}"); proposal_message = f"LLM error: {e}"; error_msg = proposal_message
        return {"messages": messages + [AIMessage(content=proposal_message)], "proposed_features": proposed_features_dict, "feature_report": "", "error_message": error_msg}

    def eng_feature(self, state: FeatureEngAgentState) -> Dict[str, Any]:
        """
        Node function: Attempts to create the proposed features using df.eval()
        and adds them to the dataframe copy in the state.
        FIX: Explicitly pass numpy functions to eval scope.
        """
        print("\nExecuting Engineer Feature Node...")
        dataset = state.get("dataset"); proposed_features = state.get("proposed_features", {});
        messages = state.get("messages", []); grams = state.get("grams", {}); error_msg = state.get("error_message")
        if dataset is None:
            print("Error: Dataset missing.");
            return {"feature_report": "Error: Dataset missing.", "messages": messages + [AIMessage(content="FE skipped: Dataset missing.")], "error_message": "Dataset missing."}

        temp_df = dataset.copy()
        original_columns = set(grams.get('original_columns', list(dataset.columns)))
        report_lines = []; successfully_added = []; failed_eval = []

        if not proposed_features:
            report_lines.append("No features proposed.")
        else:
            print(f"  Attempting: {list(proposed_features.keys())}")
            # FIX: Define the scope for eval, explicitly including np and common functions
            eval_globals = {'np': np, '__builtins__': {}} # Keep builtins empty for safety if possible
            eval_locals = {
                'log1p': np.log1p,
                'exp': np.exp,
                'log': np.log,
                'sqrt': np.sqrt,
                'abs': np.abs,
                # Add other functions the LLM might reasonably use
            }

            for name, formula in proposed_features.items():
                if not isinstance(name, str) or not name.isidentifier() or not isinstance(formula, str) or not formula:
                    report_lines.append(f"Skipping '{name}': Invalid name/formula."); failed_eval.append(str(name)); continue
                if name in temp_df.columns:
                    report_lines.append(f"Skipping '{name}': Exists."); continue
                try:
                    print(f"  Evaluating: {name} = {formula}")
                    # Pass the defined globals and locals to eval
                    # engine='python' allows more complex operations
                    new_col_data = temp_df.eval(formula, engine='python', global_dict=eval_globals, local_dict=eval_locals)

                    if isinstance(new_col_data, (pd.Series, np.ndarray)) and len(new_col_data) == len(temp_df):
                        temp_df[name] = new_col_data; report_lines.append(f"Success: '{name}'"); successfully_added.append(name)
                    else:
                        report_lines.append(f"Failed '{name}': Bad shape/type ({type(new_col_data)})."); failed_eval.append(name)
                except Exception as e:
                    err_type = type(e).__name__; report_lines.append(f"Failed '{name}': {err_type} - {e}\n  Formula: {formula}")
                    print(f"  Failed '{name}': {err_type} - {e}"); failed_eval.append(name)
                    if error_msg is None: error_msg = f"Failed to engineer '{name}'." # Set first error encountered

        final_report = "Feature Engineering Report:\n" + "\n".join(report_lines); print(final_report)
        return {
            "dataset": temp_df, "feature_report": final_report,
            "messages": messages + [AIMessage(content=f"FE attempt complete. Added: {successfully_added}. Failed: {failed_eval}.") ],
            "error_message": error_msg
        }

    def eng(self, df: pd.DataFrame, description_state: dict) -> Dict[str, Any] | None:
        print("\n=== Starting Feature Engineering Agent ===")
        if not isinstance(df, pd.DataFrame) or df.empty: print("Error: Invalid DataFrame."); return None
        if not isinstance(description_state, dict): print("Error: Invalid description_state."); return None
        thread_id = str(uuid.uuid4()); cfg = {"configurable": {"thread_id": thread_id}}
        initial_grams = description_state.get("grams", {});
        if 'original_columns' not in initial_grams: initial_grams['original_columns'] = set(df.columns)
        initial_state_dict = {
            "dataset": df.copy(), "messages": description_state.get("messages", []), "grams": initial_grams,
            "cmsg": description_state.get("cmsg"), "proposed_features": {}, "feature_report": "",
            "error_message": description_state.get("error_message")
        }
        final_state_values = None
        try:
            for event in self.graph.stream(initial_state_dict, config=cfg, stream_mode="values"):
                 final_state_values = event; last_node = list(event.keys())[-1]; print(f"  Node '{last_node}' executed.")
            print("\nFeature Engineering graph execution finished.")
        except Exception as e:
            print(f"\nError during FE graph execution: {e}"); traceback.print_exc()
            try: final_state_values = self.graph.get_state(cfg).values
            except Exception as state_err: print(f"Could not retrieve state: {state_err}"); final_state_values = {"error": str(e)}
        return final_state_values


# ==============================================================================
# Clustering Agent Class (Unchanged from v4)
# ==============================================================================
class ClusteringAgent:
    def __init__(self):
        g = StateGraph(ClusteringState)
        g.add_node("prepare_clustering_data", self.prepare_clustering_data)
        g.add_node("run_kmeans", self.run_kmeans_node)
        g.add_node("run_gmm", self.run_gmm_node)
        g.add_node("run_agglomerative", self.run_agglomerative_node)
        g.add_node("run_dbscan", self.run_dbscan_node)
        g.add_node("evaluate_clusters", self.evaluate_clusters_node)
        g.add_node("visualize_clusters", self.visualize_clusters_node)
        g.add_node("error_handler", self.error_node)
        g.set_entry_point("prepare_clustering_data")
        g.add_conditional_edges("prepare_clustering_data", self.should_continue, {"continue": "run_kmeans", "error_end": "error_handler"})
        g.add_edge("run_kmeans", "run_gmm")
        g.add_edge("run_gmm", "run_agglomerative")
        g.add_edge("run_agglomerative", "run_dbscan")
        g.add_conditional_edges("run_dbscan", self.should_continue, {"continue": "evaluate_clusters", "error_end": "error_handler"})
        g.add_conditional_edges("evaluate_clusters", self.should_continue, {"continue": "visualize_clusters", "error_end": "error_handler"})
        g.add_conditional_edges("visualize_clusters", self.should_continue, {"continue": END, "error_end": "error_handler"})
        g.add_edge("error_handler", END)
        memory = MemorySaver(serde=pickle)
        self.graph = g.compile(checkpointer=memory)
        print("Clustering Agent Graph compiled.")

    def prepare_clustering_data(self, state: ClusteringState) -> Dict[str, Any]:
        print("\n--- Node: Preparing Clustering Data ---")
        X = state.get('input_data_for_clustering'); feature_names = state.get('feature_names'); error_msg = state.get("error_message")
        if X is None or not isinstance(X, np.ndarray) or X.shape[0] < 2 or X.shape[1] < 1:
            print("  Error: Invalid data for clustering."); error_msg = "Invalid data for clustering."
            return {"input_data_for_clustering": None, "error_message": error_msg}
        print(f"  Data ready. Shape: {X.shape}")
        return {
            "input_data_for_clustering": X, "feature_names": feature_names, "clustering_results": {},
            "evaluation_summary": None, "best_method": None, "best_labels": None, "best_model": None,
            "visualization_base64": None, "error_message": error_msg
        }

    def run_kmeans_node(self, state: ClusteringState) -> Dict[str, Any]:
        print("--- Node: Running K-Means ---")
        X = state.get('input_data_for_clustering'); results = state.get('clustering_results', {}); error_msg = state.get("error_message")
        if X is None: return {"error_message": error_msg or "K-Means skipped."}
        k_range = range(2, 11); best_score, best_k, best_labels_km, best_model_km = -1.1, -1, None, None
        print("  Tuning K:");
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X); labels = kmeans.labels_
                if len(np.unique(labels)) >= 2:
                    score = silhouette_score(X, labels); print(f"    K={k}, Score={score:.4f}")
                    if score > best_score: best_score, best_k, best_labels_km, best_model_km = score, k, labels, kmeans
                else: print(f"    K={k}, Score: < 2 clusters.")
            except Exception as e: print(f"    K={k}, Error: {e}")
        if best_k != -1:
            print(f"  Best K: {best_k} (Score: {best_score:.4f})")
            results['KMeans'] = {'model': best_model_km, 'labels': best_labels_km, 'score': best_score, 'params': {'n_clusters': best_k}, 'noise_ratio': 0.0}
        else: print("  K-Means failed."); results['KMeans'] = {'score': -1.1}
        return {"clustering_results": results, "error_message": error_msg}

    def run_gmm_node(self, state: ClusteringState) -> Dict[str, Any]:
        print("--- Node: Running GMM ---")
        X = state.get('input_data_for_clustering'); results = state.get('clustering_results', {}); error_msg = state.get("error_message")
        if X is None: return {"error_message": error_msg or "GMM skipped."}
        n_components_range = range(2, 11); best_score, best_n, best_labels_gmm, best_model_gmm = -1.1, -1, None, None
        print("  Tuning Components:")
        for n in n_components_range:
            try:
                gmm = GaussianMixture(n_components=n, random_state=42).fit(X); labels = gmm.predict(X)
                if len(np.unique(labels)) >= 2:
                    score = silhouette_score(X, labels); print(f"    Components={n}, Score={score:.4f}")
                    if score > best_score: best_score, best_n, best_labels_gmm, best_model_gmm = score, n, labels, gmm
                else: print(f"    Components={n}, Score: < 2 clusters.")
            except Exception as e: print(f"    Components={n}, Error: {e}")
        if best_n != -1:
            print(f"  Best Components: {best_n} (Score: {best_score:.4f})")
            results['GMM'] = {'model': best_model_gmm, 'labels': best_labels_gmm, 'score': best_score, 'params': {'n_components': best_n}, 'noise_ratio': 0.0}
        else: print("  GMM failed."); results['GMM'] = {'score': -1.1}
        return {"clustering_results": results, "error_message": error_msg}

    def run_agglomerative_node(self, state: ClusteringState) -> Dict[str, Any]:
        print("--- Node: Running Agglomerative Clustering ---")
        X = state.get('input_data_for_clustering'); results = state.get('clustering_results', {}); error_msg = state.get("error_message")
        if X is None: return {"error_message": error_msg or "Agglomerative skipped."}
        k_agglo = results.get('GMM', {}).get('params', {}).get('n_components') or results.get('KMeans', {}).get('params', {}).get('n_clusters') or 3
        print(f"  Using n_clusters = {k_agglo}")
        try:
            agglo = AgglomerativeClustering(n_clusters=k_agglo).fit(X); labels = agglo.labels_
            score = -1.1
            if len(np.unique(labels)) >= 2: score = silhouette_score(X, labels); print(f"  Score: {score:.4f}")
            else: print("  Score: < 2 clusters.")
            results['Agglomerative'] = {'model': agglo, 'labels': labels, 'score': score, 'params': {'n_clusters': k_agglo}, 'noise_ratio': 0.0}
        except Exception as e: print(f"  Agglomerative failed: {e}"); results['Agglomerative'] = {'score': -1.1}
        return {"clustering_results": results, "error_message": error_msg}

    def run_dbscan_node(self, state: ClusteringState) -> Dict[str, Any]:
        print("--- Node: Running DBSCAN ---")
        X = state.get('input_data_for_clustering'); results = state.get('clustering_results', {}); error_msg = state.get("error_message")
        if X is None: return {"error_message": error_msg or "DBSCAN skipped."}
        eps_val, min_samples_val = 0.5, 5; print(f"  Using eps={eps_val}, min_samples={min_samples_val}")
        try:
            dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val).fit(X); labels = dbscan.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0); n_noise = list(labels).count(-1)
            noise_ratio = n_noise / len(labels) if len(labels) > 0 else 0
            print(f"  Found {n_clusters} clusters, {n_noise} noise points (Ratio: {noise_ratio:.2%}).")
            score = -1.1; non_noise_mask = (labels != -1)
            if n_clusters >= 2 and np.sum(non_noise_mask) > 1:
                 unique_cluster_labels = np.unique(labels[non_noise_mask])
                 if len(unique_cluster_labels) >= 2:
                     try: score = silhouette_score(X[non_noise_mask], labels[non_noise_mask]); print(f"  Silhouette (non-noise): {score:.4f}")
                     except ValueError as e: print(f"  Silhouette calc error: {e}")
                 else: print("  Silhouette not calculated (all non-noise in one cluster).")
            else: print("  Silhouette not calculated (< 2 clusters or points).")
            results['DBSCAN'] = {'model': dbscan, 'labels': labels, 'score': score, 'params': {'eps': eps_val, 'min_samples': min_samples_val}, 'noise_ratio': noise_ratio}
        except Exception as e: print(f"  DBSCAN failed: {e}"); results['DBSCAN'] = {'score': -1.1, 'noise_ratio': 1.0}
        return {"clustering_results": results, "error_message": error_msg}

    def evaluate_clusters_node(self, state: ClusteringState) -> Dict[str, Any]:
        print("--- Node: Evaluating Clustering Results ---")
        results = state.get('clustering_results'); error_msg = state.get("error_message")
        if not results: return {"error_message": error_msg or "Eval skipped."}
        best_method_name, best_score, best_labels_eval, best_model_eval = None, -np.inf, None, None
        MAX_ACCEPTABLE_NOISE_RATIO = 0.50; print("  Evaluating methods:"); evaluation = {}
        for method in sorted(results.keys()):
            result = results.get(method, {}); score = result.get('score', -1.1); noise_ratio = result.get('noise_ratio', 0.0)
            params = result.get('params', {}); model = result.get('model'); labels = result.get('labels')
            is_valid = (score >= -1.0); is_best_candidate = is_valid
            log_prefix = f"    Method: {method},"; log_suffix = f"Score={score:.4f}, Noise Ratio={noise_ratio:.2%}"
            if method == 'DBSCAN' and noise_ratio > MAX_ACCEPTABLE_NOISE_RATIO: print(f"{log_prefix} Raw {log_suffix} -> Penalized."); is_best_candidate = False
            elif is_valid: print(f"{log_prefix} {log_suffix}")
            else: print(f"{log_prefix} Invalid Score."); is_best_candidate = False
            evaluation[method] = {'score': score, 'noise_ratio': noise_ratio, 'params': params, 'is_valid_candidate': is_best_candidate}
            if is_best_candidate and score > best_score: best_score, best_method_name, best_labels_eval, best_model_eval = score, method, labels, model
        if best_method_name:
            print(f"\n  Best Method: {best_method_name} (Score: {best_score:.4f})")
            return {"evaluation_summary": evaluation, "best_method": best_method_name, "best_labels": best_labels_eval, "best_model": best_model_eval, "error_message": error_msg}
        else: print("\n  Eval failed: No valid method."); return {"evaluation_summary": evaluation, "error_message": error_msg or "No valid clustering method."}

    def visualize_clusters_node(self, state: ClusteringState) -> Dict[str, Any]:
        print("--- Node: Visualizing Best Cluster ---")
        X = state.get('input_data_for_clustering'); labels = state.get('best_labels'); method_name = state.get('best_method'); error_msg = state.get("error_message")
        if X is None or labels is None or method_name is None: return {"error_message": error_msg or "Viz skipped."}
        try:
            pca_viz = PCA(n_components=2, random_state=42); X_pca = pca_viz.fit_transform(X); print(f"  PCA done.")
            plt.figure(figsize=(10, 7)); df_plot = pd.DataFrame({'PCA1': X_pca[:, 0], 'PCA2': X_pca[:, 1], 'Cluster': labels})
            unique_labels = sorted(df_plot['Cluster'].unique()); n_colors = len(unique_labels) - (1 if -1 in unique_labels else 0)
            palette = sns.color_palette("viridis", n_colors=max(1, n_colors)); color_map = {label: palette[i] for i, label in enumerate(l for l in unique_labels if l != -1)}
            if -1 in unique_labels: color_map[-1] = (0.5, 0.5, 0.5)
            sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette=color_map, data=df_plot, legend='full', s=50, alpha=0.7)
            plt.title(f'Best Clustering ({method_name}) - PCA Viz'); plt.xlabel('PC1'); plt.ylabel('PC2')
            plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout()
            buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); plt.close(); buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8'); print(f"  Viz generated.")
            return {"visualization_base64": img_base64, "error_message": error_msg}
        except Exception as e: print(f"  Viz error: {e}"); plt.close(); return {"visualization_base64": None, "error_message": error_msg or f"Viz failed: {e}"}

    def should_continue(self, state: ClusteringState) -> str:
        if state.get("error_message"): print(f"--- Condition: Error ('{state['error_message']}'), routing to error ---"); return "error_end"
        else: print(f"--- Condition: No error, continuing ---"); return "continue"

    def error_node(self, state: ClusteringState) -> Dict[str, Any]:
        print("--- Node: Handling Error State ---")
        print(f"  Error: {state.get('error_message', 'Unknown')}")
        return {} # Return state as is

    def cluster(self, preprocessed_data: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, Any] | None:
        print("\n=== Starting Clustering Agent ===")
        if not isinstance(preprocessed_data, np.ndarray) or preprocessed_data.shape[0] < 2: print("Error: Invalid data."); return None
        thread_id = str(uuid.uuid4()); cfg = {"configurable": {"thread_id": thread_id}}
        initial_state_dict = {
            "input_data_for_clustering": preprocessed_data, "feature_names": feature_names, "clustering_results": {},
            "evaluation_summary": None, "best_method": None, "best_labels": None, "best_model": None,
            "visualization_base64": None, "error_message": None
        }
        final_state_values = None
        try:
            for event in self.graph.stream(initial_state_dict, config=cfg, stream_mode="values"):
                 final_state_values = event; last_node = list(event.keys())[-1]; print(f"  Node '{last_node}' executed.")
            print("\nClustering Agent graph execution finished.")
        except Exception as e:
            print(f"\nError during Clustering Agent graph execution: {e}"); traceback.print_exc()
            try: final_state_values = self.graph.get_state(cfg).values
            except Exception as state_err: print(f"Could not retrieve state: {state_err}"); final_state_values = {"error": str(e)}
        # Clean up non-serializable objects before returning
        if final_state_values: return self._cleanup_state_dict(final_state_values)
        else: return None

    def _cleanup_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Internal helper to make state serializable."""
        print("Cleaning up final clustering state...")
        safe_state = {}
        for key in ["best_method", "error_message", "visualization_base64", "evaluation_summary", "feature_names"]:
            if key in state_dict and state_dict[key] is not None: safe_state[key] = state_dict[key]
        if "clustering_results" in state_dict and state_dict["clustering_results"]:
            safe_results = {}
            for method, data in state_dict["clustering_results"].items():
                safe_data = {k: v for k, v in data.items() if k not in ['model', 'labels']}
                if 'labels' in data and data['labels'] is not None:
                    try: safe_data['labels_preview'] = data['labels'][:10].tolist()
                    except: pass
                safe_results[method] = safe_data
            safe_state["clustering_results"] = safe_results
        if "best_labels" in state_dict and state_dict["best_labels"] is not None:
            try: safe_state["best_labels_preview"] = state_dict["best_labels"][:10].tolist()
            except: pass
        print("Final state cleaned.")
        return safe_state
