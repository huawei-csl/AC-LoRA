import logging
from glob import glob
from tqdm import tqdm
import os
import json


class Grader:
    def __init__(self, config):
        self.grading_model = config.grading.grading_model
        self.grading_type = config.grading.type
        self.files_to_grade = [
            x
            for xs in [
                glob(config.data.data_dir_root + file)
                for file in config.data.evaluation_data.split(",")
            ]
            for x in xs
            if "grade" not in x
        ]
        self.repetitions = config.grading.repetitions
        self.grading_temp = config.grading.temperature
        self.seed = config.meta.seed

        self.base_url = config.grading.url
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.total_cost = 0

    def openai_chat(self, messages, configs):
        return self.openai.chat.completions.create(
            messages=messages, model=self.grading_model, **configs
        )

    def ollama_chat(self, messages, configs):
        import ollama

        return ollama.chat(messages=messages, model=self.grading_model, **configs)

    def grade_api(self):
        from openai import OpenAI

        self.openai = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.generation_config = {
            "frequency_penalty": 0,
            "max_completion_tokens": 1024,
            "n": 1,
            "seed": self.seed,
            "temperature": self.grading_temp,
        }
        self.model = self.openai_chat
        self.extract_answer = lambda x: x.choices[0].message.content

        for file in self.files_to_grade:
            # we use the same file just add _grade to it
            output_file = (
                file[:-5]
                + "_grade_api_"
                + self.grading_model.replace("/", "_")
                + ".json"
            )
            with open(file, "r") as fp:
                data_to_grade = json.load(fp)
            # first row is the metadata k, fetch_k and threshold
            graded_results = self.grading_loop(data_to_grade=data_to_grade)

            with open(output_file, "w+") as f:
                json.dump(graded_results, f)
        print("TOTAL EXPECTED COST: ", self.total_cost)

    def _grade(self, query, reference_answer, long_reference_answer, generated_answer):
        messages = [
            {
                "role": "system",
                "content": "Evaluate how well the Generated Answer matches the Reference Answer or the detailed reference answer for the given Query. Be strict: Names, dates, and specific details must be exact to be correct. Additional facts that are not in the Reference Answer do not affect the score unless they contradict the Reference Answer, in which case the score should decrease. If a name, date, or key fact is incorrect, the score must be 1, regardless of other details. Assign a score from 1 to 5 based on accuracy, completeness, and relevance: 5 = Identical meaning, all details correct. 4 = Mostly correct, with only minor wording variations but the same meaning. 3 = Partially correct, with some missing or incorrect details. 2 = Weak relevance, with significant errors or omissions. 1 = Incorrect or unrelated. Input Format: Query: {query}\n Reference Answer: {reference_answer}\nGenerated Answer: {generated_answer}\n Output Format: Explanation: [Brief reason for the score] Score: [1-5]",
            },
            {
                "role": "user",
                "content": f"Query: {query}\n Reference Answer: {reference_answer}\nDetailed reference Answer: {long_reference_answer}\n Generated Answer: {generated_answer}",
            },
        ]
        grade = []
        if not generated_answer:
            # for safety should not happen
            return
        for _ in range(self.repetitions):
            while len(grade) < 2:  # correct format?
                response = self.model(
                    messages=messages,
                    configs=self.generation_config,
                )
                if type(response) != dict:
                    self.total_cost += response.usage.estimated_cost
                answ = self.extract_answer(response)
                grade = answ.split("Score:")
            return grade[1].strip(), grade[0].strip()

    def grading_loop(self, data_to_grade):
        graded_results = []
        bar = tqdm(data_to_grade[1:])
        for idx, d in enumerate(bar):
            query = d["messages"][0]["content"]
            reference_answer = d["answer"]
            long_reference_answer = d["long_answer"]

            if "prediction" in d.keys():
                generated_answer = d["prediction"][0][0].split("assistant\n\n")[-1]
            elif "generated" in d.keys():
                generated_answer = d["generated"][0].split("assistant\n\n")[-1]
            else:
                generated_answer = d["predictions"][
                    0
                ]  # TODO might be more? Just take the first atm

            grades, explaination = [], []

            res = self._grade(
                query=query,
                reference_answer=reference_answer,
                long_reference_answer=long_reference_answer,
                generated_answer=generated_answer,
            )
            # in case we make multiple grade runs
            if res:
                grades.append(res[0])
                explaination.append(res[1])
            # just for hint - rebuttal
            hint_grades, hint_explaination = [], []
            if "hint_prediction" in d.keys() and d["hint_prediction"]:
                generated_answer = d["hint_prediction"][0][0].split("assistant\n\n")[-1]
                res = self._grade(
                    query=query,
                    reference_answer=reference_answer,
                    long_reference_answer=long_reference_answer,
                    generated_answer=generated_answer,
                )
                if res:
                    hint_grades.append(res[0])
                    hint_explaination.append(res[1])
            graded_results.append(
                {
                    **d,
                    "grade": grades,
                    "explanation": explaination,
                    "hint_grades": hint_grades,
                    "hint_explanation": hint_explaination,
                }
            )
            bar.set_postfix_str(f"current_expected_cost={self.total_cost}")
        return graded_results

    def grade_hf(self):
        # Not implemented - we only used ollama.
        return

    def grade_ollama(self):
        self.generation_config = {"options": {"temperature": self.grading_temp}}
        self.model = self.ollama_chat
        self.extract_answer = lambda x: x["message"]["content"]

        for file in self.files_to_grade:
            # we use the same file just add _grade to it
            output_file = (
                file[:-5]
                + "_grade_ollama_"
                + self.grading_model.replace("/", "_")
                + ".json"
            )
            with open(file, "r") as fp:
                data_to_grade = json.load(fp)
            # first row is the metadata k, fetch_k and threshold
            graded_results = self.grading_loop(data_to_grade=data_to_grade)

            with open(output_file, "w+") as f:
                json.dump(graded_results, f)

    def grade_flan(self):
        # ATM computed directly at the plot generation
        pass

    def grade(self):
        if self.grading_type == "grading":
            self.grade_ollama()
        if self.grading_type == "grading_api":
            self.grade_api()
        else:
            logging.error(
                "Not implemented. We graded Flan directly in the plot scripts."
            )
            return
