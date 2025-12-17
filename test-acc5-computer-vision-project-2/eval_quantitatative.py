from LLM import LLModel
import pandas as pd
import time
import ast
from LLM import LLModel



def process_quantities_sequential(model_instance, dataframe, delay=0, chunk_size=2, save_path=None):
    """
    Processes a batch of prompts sequentially with periodic saving and stores responses in the dataframe.
    """
    results = []
    for i, row in dataframe.iterrows():
        image_path = row['path']
        response_truth = row['true_answer']
        response_generated = row['model_answer']
        question = row['question']
        print(f"Processing row {i+1}/{len(dataframe)}...")

        try:
            quantity_truth, quantity_generated = get_quantitites(model_instance, question, response_truth, response_generated)
        except Exception as e:
            print(f"Error generating response: {e}")
            quantity_truth, quantity_generated = None, None

        dataframe.at[i, 'Quantity_truth'] = quantity_truth  # Store response in the new column
        dataframe.at[i, 'Quantity_generated'] = quantity_generated  # Store response in the new column

        if save_path and (i + 1) % chunk_size == 0:
            partition_start = i + 1 - chunk_size
            dataframe.iloc[:i+1].to_csv(f"{save_path}/partition_{partition_start}_{i+1}.csv", index=False)
            print(f"Saved partition {partition_start} to {i+1}")

        time.sleep(delay)

    return dataframe

def get_prompt(question, answer_A, answer_B):
    prompt = f"""
    We are testing the spatial reasoning capabilities of VLM models. 
    You will be given a question regarding an image. Then, you will be given the answer to the questions given by two models  A and B.
    Both answers provide a quantitative response to the question. 
    Your task is to extract the relevant quantities from both of the answers.
    The quantitities must be in the same units, if they are not, you must convert them to the same units.
    Your output must be a list of both quantities in meters. If you can not extract the quantities, please return None
    
    Example 1:
    Question: "How far is the yellow finger from the silver can?"
    Answer A: "About one and a half meters"
    Answer B: "The yellow finger is 300 cms away from the blue container and 200cms away from the silver can"
    Output: [1.5, 2.0]
    
    Example 2:
    Question: "How wide is the path between the sofa and the table?"
    Answer A: "The path is 1.56m wide"
    Answer B: "The path between the table and the table is 2.3m wide. However, there is much more space in the path between the sofa and the table, about 5m."
    Output: [1.56, 5.0]
    
    Example 3: 
    Question: "How long is the table?"
    Answer A: "The table is 2m long"
    Answer B: "The chair is 200cm long"
    Output: [2.0, None]
    
    
    Ok, the question is: {question}
    The answer from model A is: {answer_A}
    The answer from model B is: {answer_B}
    
    Please provide the quantities in meters. No greetings or comments. Just provide the quantities.
    """

    return prompt

def get_quantitites(llm, question, answer_A, answer_B):
    prompt = get_prompt(question, answer_A, answer_B)
    response = llm.generate_response(prompt)
    actual_list = ast.literal_eval(response)
    return actual_list[0], actual_list[1]    


def save_results_to_csv(results, filename):
    """Saves results to a CSV file."""
    try:
        # Convert the results list of dictionaries into a pandas DataFrame
        df = pd.DataFrame(results)
        # Save the DataFrame to a CSV file
        df.to_csv(filename, index=False, encoding="utf-8")
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results to {filename}: {e}")


def get_statistics(updated_df):
    quantities_df = updated_df[['Quantity_truth', 'Quantity_generated']].dropna()
    
    # Calculate within range percentage
    within_range = quantities_df.apply(
        lambda row: 0.5 * row['Quantity_truth'] <= row['Quantity_generated'] <= 2 * row['Quantity_truth'], axis=1
    )
    within_range_percentage = within_range.mean() * 100
    
    # Calculate deviations
    deviations = abs(quantities_df['Quantity_generated'] - quantities_df['Quantity_truth']) / quantities_df['Quantity_truth']
    
    median_deviation = deviations.median()
    average_deviation = deviations.mean()
    
    return {
        "within_range_percentage": within_range_percentage,
        "median_deviation": median_deviation,
        "average_deviation": average_deviation
    }

def quantitative_evaluation(model_sequential, model_name):  
      
    # 1. Load DataFrame with image paths and descriptions (Replace with actual loading logic)
    path_to_answers_open_source = f"src/output_files/{model_name}/generated_answers_open_source/all.csv"
    path_quantities_csv = f"src/output_files/{model_name}/generated_quantities/all.csv"
    path_stats = f"src/output_files/{model_name}/quantitative_evaluation/stats.txt"
    save_path = f"src/output_files/{model_name}/generated_quantities"
    
    df = pd.read_csv(path_to_answers_open_source)
  
    updated_df = process_quantities_sequential(model_sequential, df, delay=0, chunk_size=2, save_path=save_path)
    
    # Save the updated dataframe to a CSV file
    updated_df.to_csv(path_quantities_csv, index=False, encoding="utf-8")
    
    statistics = get_statistics(updated_df)
    
    # Save statistics to a file
    with open(path_stats, "w") as file:
        for key, value in statistics.items():
            file.write(f"{key}: {value}\n")

if __name__ == "__main__":
    model_name = "mock"
    model_sequential = LLModel()  # Pass config and settings
    quantitative_evaluation(model_sequential, model_name)
