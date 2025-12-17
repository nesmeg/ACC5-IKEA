import pandas as pd
import time
import ast

from LLM import LLModel



def process_batch_sequential(model_instance, dataframe, delay=0, chunk_size=500):
    """
    Processes a batch of prompts sequentially with periodic saving and stores responses in the dataframe.
    """
    paths= []
    descriptions= []
    questions= []
    true_answer= []
    type_questions= []
    completed_flags= []
    
    for i, row in dataframe.iterrows():
        image_path = row['path']
        description = row['description']
        print(f"Processing row {i+1}/{len(dataframe)}...")

        #try:
        prompt = get_prompt(description)
        response_text = model_instance.generate_response(prompt)
        list_response = ast.literal_eval(response_text)
        completed_flag = True
        for response in list_response:
            paths.append(image_path)
            descriptions.append(description)
            questions.append(response[0])
            true_answer.append(response[1])
            type_questions.append(response[2])
            completed_flags.append(completed_flag)
            
        #except Exception as e:
        #    print(f"Error generating response: {e}")
            
        
        # Store response in the new column
    df = pd.DataFrame({
        "path": paths,
        "description": description,
        "question": questions,
        "true_answer": true_answer,
        "type_question": type_questions,
    })

    return df

def get_prompt(description):
    prompt = f"""
    We are testing the spatial reasoning capabilities of VLM models. 
    You will be given a description of an image. The description will describe entities in the image and their spatial relationships.
    Your goal is to generate set of qualititive and quantitative questions that can be answered with the description. 
    Your qualitative questions should ask about semantic spatial relationships between entities in the image, for example: "Where is the cat positioned with respect to the table (on top, underneath, right to the left, ...)?"
    Your quantitative questions should ask about the distance between two entities in the image, for example: "How far in meters is the cat from the table?"
    Those are the only two types of questions you should generate.
    You should not ask questions about dimensions of entities or any other type of information.
    You neeed to avoid spatial information leakage in the question.
    The questions should be clear, concise, and relevant to the image content.
    Reminder: do only generate questions that can be answered with the description.

    You need to generate the questions such that there is only one possible answer to it.
    You need to also return the answer to the question and whether it is qualitative or quantitative. The answer should be a piece of information that can be extracted from the description.
    Your answer **must be** a Python list of tuples, formatted exactly as:

    [('Question', 'Answer', 'qualitative/quantitative'), ('Question', 'Answer', 'qualitative/quantitative')]

    Example description:
    
    The bed with a pink comforter and pillows is positioned against the left wall, 
    about 3 meters wide. On the left side of the bed, 
    there is a red ball and a small toy train, both about 0.5 meters tall.  The red ball is about one meter away from the bed.
    The white dresser with a mirror is placed against the right wall. The dresser is a meter and a half to the right of the bed. 
    Between the bed and the dresser, there are two windows, one on the left and one on the right side of the room, each about 1 meter wide. 
    The walls are painted pink with white trim, and the floor is carpeted.
    Please ensure that the questions do not ask about anything other than spatial information.
    
    Example output:[
    ("Where is the bed positioned with respect to the left wall?", "Against the left wall", "qualitative"), ("Where is the dresser positioned with respect to the right wall?", "Against the right wall", "qualitative"), ("Where is the red ball positioned with respect to the bed?", "On the left side of the bed", "qualitative"),("Where is the toy train positioned with respect to the bed?", "On the left side of the bed", "qualitative"),("Where are the windows positioned with respect to the bed?", "Between the bed and the dresser", "qualitative"),("How far is the red ball from the bed?", "One meter", "quantitative"),("How far is the dresser from the bed?", "One and a half meters", "quantitative"),("How wide is the space occupied by the two windows together?", "Two meters", "quantitative")]

    Your output must be a python list of tuples.
    It is important that you do not repeat questions.
    Try to generate 8 questions, only if you can find 8 distinct ones. DO NOT REPEAT QUESTIONS. If you can not find 8 distinct questions, you ust provide less.
    Ok, the description is: {description}
    Please provide your answer. Do not provide any greetings, or comments. Just provide the answer.
    """
    return prompt


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


def generate_questions(path_to_csv, model_name):    
    # 1. Load DataFrame with image paths and descriptions (Replace with actual loading logic)
    df = pd.read_csv(path_to_csv)
    
    nr_calls_between_saves = 3
    model_sequential = LLModel()  # Pass config and settings

    # Process the dataframe in partitions
    for start in range(0, len(df), nr_calls_between_saves):
        end = start + nr_calls_between_saves
        df_partition = df.iloc[start:end]
        updated_df_partition = process_batch_sequential(model_sequential, df_partition, delay=0, chunk_size=500)
        
        # Save the updated partition to a CSV file
        updated_df_partition.to_csv(f"src/intermediate_output_files/{model_name}/generated_questions/partition_{start}_{end}.csv", index=False)

        if start == 0:
            total_updated_df = updated_df_partition
        else:
            total_updated_df = pd.concat([total_updated_df, updated_df_partition])
        
        # Print progress
        print(f"Processed partition {start} to {end}")
    # Print Results (example from updated DataFrame - limited to first 10 rows)
    print("\nFirst 10 Rows with Generated Questions:")
    
    return total_updated_df


if __name__ == "__main__":
    model_name = 'mock'
    df_with_questions = generate_questions("src/output_files/mock/ground_truth_SpaceMantis-8B.csv", model_name=model_name)
    df_with_questions.to_csv("src/utput_files/mock/generated_questions/all.csv", index=False)
