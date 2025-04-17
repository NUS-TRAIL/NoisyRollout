import re
import os
import google.generativeai as genai

def get_gpt4_ICE():
    example_1 = """
Hint: Please answer the question requiring an integer answer and provide the final value,
e.g., 1, 2, 3, at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
Extracted answer: 14
"""

    example_2 = """
Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value,
e.g., 1.2, 1.3, 1.4, at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
Extracted answer: 0.6
"""

    example_3 = """
Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value,
e.g., 1.23, 1.34, 1.45, at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: 1.45
"""

    example_4 = """
Hint: Please answer the question requiring a Python list as an answer and provide the final list,
e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The line graph saw its maximum peak between 2007 and 2008.\n
Extracted answer: [2007, 2008]
"""

    example_5 = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
Extracted answer: B
"""

    return [example_1, example_2, example_3, example_4, example_5]


def get_gpt4_score_ICE():
    example_1 = """
[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer]: Extracted Answer: \\((-2, 1)\\)
Judgement: 0
""" # noqa

    example_2 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer]: B:2\u221a{{3}}
Judgement: 0
""" # noqa

    example_3 = """
[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer]: Range: \\((-4, 1]\\)
Judgement: 0
""" # noqa

    example_4 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer]: null
Judgement: 0
""" # noqa

    example_4 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\n(A):2\n(B):2\u221a{{3}}\n(C):\u221a{{3}}\n(D):2\u221a{{2}}
[Standard Answer]: (C)
[Model_answer]: C
Judgement: 1
""" # noqa
    return [example_1, example_2, example_3, example_4]



def get_gpt4_extract_ICE():
    example_1 = """
1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: (-2, 1)
""" # noqa

    example_2 = """
2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D
""" # noqa

    example_3 = """
3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)
""" # noqa

    example_4 = """
4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: null
""" # noqa

    example_5 = """
5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3
""" # noqa

    example_6 = """
6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: f(x) = -x^2 - 2x + 1
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6]


def build_score_prompt(question, extract, answer):
    task_description = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.\n\n
""" # noqa
    demo_prompt = task_description
    examples = get_gpt4_score_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
    Please output the judgement score directly with no explanation.
    [Question]: {question}
    [Standard Answer]: {answer}
    [Model_answer]: {extract}
    Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'

    return full_prompt


def build_extract_prompt(prediction, question):
    task_description = """
Please read the following example.
Then output the answer extracted from the model response directly. No "Extracted answer:" in your answer.\n
"""
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model respone: ' + prediction
    prompt += 'Extracted answer:'
    return prompt

def build_wemath_extract_prompt(extraction: str, question: str) -> str:
    prompt = f"""You are evaluating answers to math questions. Extract the final answer from the text.

Question: {question}

Model's solution:
{extraction}

Extract the final answer as a single letter (A, B, C, D, or E) without any explanation or other text.
If the final answer is not clear, make your best determination based on the reasoning provided.
Your output should be ONLY the letter corresponding to the answer choice.
"""
    return prompt

def build_mathverse_extract_prompt(prediction):
    task_description = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.\n\n
""" # noqa
    demo_prompt = task_description
    examples = get_gpt4_extract_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"Model response: '{prediction}'\nExtracted Answer: "
    full_prompt = f'{demo_prompt}7.\n{test_prompt}'

    return full_prompt

def extract_boxed_answer(text):
    """Extract the last boxed answer from generated text, if present."""
    boxed_matches = re.findall(r'\\boxed{([^}]+)}', text)
    if boxed_matches:
        return boxed_matches[-1].strip(), True # Return the last match
    return text, False

def llm_eval_score(question, prediction, answer, dataset):
    google_api_key = os.getenv("GOOGLE_API_KEY", "") # Use default if env var is not set
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel("gemini-2.0-flash-001")

    if dataset.lower() == "mathverse":
        extracted_answer, boxed_flag = extract_boxed_answer(prediction)
        if not boxed_flag:
            extract_prompt = build_mathverse_extract_prompt(prediction)
            extracted_answer = model.generate_content(extract_prompt, generation_config={"temperature":0.0}).text

        score_prompt = build_score_prompt(question, extracted_answer, answer)
        
        for _ in range(3):
            response_text = model.generate_content(score_prompt, generation_config={"temperature":0.0}).text.strip()
            if response_text in ['0', '1']:
                return int(response_text)
        return 0.0
    
    elif dataset.lower() in ["mathvista", "mathvision"]:
        extract_prompt = build_extract_prompt(prediction, question)
        extracted_answer = model.generate_content(extract_prompt, generation_config={"temperature":0.0}).text

        # TODO: retry?
        if extracted_answer.strip() == answer:
            return 1.0
        else:
            return 0.0
        
    elif dataset.lower() == "wemath":
        extract_prompt = build_wemath_extract_prompt(prediction, question)
        response = model.generate_content(extract_prompt, generation_config={"temperature":0.0})
        extracted_answer = response.text.strip().upper()
        
        # TODO: retry?
        if re.match(r'^[A-G]$', extracted_answer):
            accuracy = 1.0 if extracted_answer == answer else 0.0
            return accuracy
        else:
            return 0.0