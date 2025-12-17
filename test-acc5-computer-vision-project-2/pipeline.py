import os
import pandas as pd
from question_generation import generate_questions
from eval_quantitative import quantitative_evaluation
from LLM import LLModel
from LLM_judge import LLMJudge
import ground_truth_gen
# TODO: Import correct model evaluation module
import vlm_answers_gen

GROUND_TRUTH_MODEL = "remyxai/SpaceMantis" # HuggingFace model to generate ground truth descriptions
MODEL_NAME = "deepseek-ai/Janus-Pro-7B" # HuggingFace model to evaluate

# Images directory
IMAGES_DIR = "img/test"

# Placeholder paths
GROUND_TRUTH_DESCRIPTIONS_PATH = "output_files/ground_truth_descriptions.csv"
QUESTIONS_PATH = "output_files/questions.csv"
MODEL_ANSWERS_PATH = f"output_files/model_answers_{MODEL_NAME}.csv"
QUALITATIVE_RESULTS_PATH = f"output_files/qualitative_results_{MODEL_NAME}.csv"
EVAL_RESULTS_PATH = f"output_files/eval_results_{MODEL_NAME}.csv"
QUANTITATIVE_RESULTS_PATH = f"output_files/quantitative_results_{MODEL_NAME}.csv"
STATS_PATH = f"output_files/stats_{MODEL_NAME}.txt"

def main():
    # Step 1: Load or generate ground truth descriptions
    if not os.path.exists(GROUND_TRUTH_DESCRIPTIONS_PATH):
        print("Ground truth descriptions file not found. Generating ground truth descriptions...")
        processor, model = ground_truth_gen.load_model(
            GROUND_TRUTH_MODEL,
            ground_truth_gen.MLlavaProcessor,
            ground_truth_gen.LlavaForConditionalGeneration
        )
        generation_kwargs = {
            "max_new_tokens": 1024,
            "num_beams": 1,
            "do_sample": False
        }
        ground_truth_gen.process_images(IMAGES_DIR, model, processor, generation_kwargs, GROUND_TRUTH_DESCRIPTIONS_PATH)
    else:
        print("Loading existing ground truth descriptions...")
    ground_truth_df = pd.read_csv(GROUND_TRUTH_DESCRIPTIONS_PATH)

    # Step 2: Generate questions based on ground truth descriptions
    if not os.path.exists(QUESTIONS_PATH):
        print("Questions file not found. Generating questions...")
        questions_df = generate_questions(GROUND_TRUTH_DESCRIPTIONS_PATH)
        questions_df.to_csv(QUESTIONS_PATH, index=False)
    else:
        print("Loading existing questions...")
    questions_df = pd.read_csv(QUESTIONS_PATH)

    # Step 3: Generate model answers for the questions
    # TODO: Revisar el modulo en si, no se si esta bien implementado
    if not os.path.exists(MODEL_ANSWERS_PATH):
        print("Model answers file not found. Generating model answers...")
        processor, model = vlm_answers_gen.load_model(
            MODEL_NAME,
            vlm_answers_gen.MLlavaProcessor,
            vlm_answers_gen.LlavaForConditionalGeneration
        )
        generation_kwargs = {
            "max_new_tokens": 1024,
            "num_beams": 1,
            "do_sample": False
        }
        vlm_answers_gen.process_questions(QUESTIONS_PATH, model, processor, generation_kwargs, MODEL_ANSWERS_PATH)
    else:
        print("Loading existing model answers...")
    model_answers_df = pd.read_csv(MODEL_ANSWERS_PATH)

    # Step 4: Evaluate the model answers against the ground truth using LLMJudge
    judge = LLMJudge()
    batch_data = model_answers_df.to_dict(orient='records')
    eval_results = judge.process_batch(batch_data)
    eval_results_df = model_answers_df.copy()
    eval_results_df['eval_result'] = eval_results
    eval_results_df.to_csv(EVAL_RESULTS_PATH, index=False)
    accuracy = eval_results_df['eval_result'].mean()
    print(f"{MODEL_NAME} accuracy: {accuracy}")
    eval_results_df.to_csv(EVAL_RESULTS_PATH, index=False)

    # Step 5: Perform quantitative evaluation
    model_sequential = LLModel()  # Pass config and settings
    quantitative_evaluation(model_sequential, model_name=MODEL_NAME)

if __name__ == "__main__":
    main()
