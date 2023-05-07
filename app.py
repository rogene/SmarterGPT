# This implementation takes in a question (with an assumption of relative complexity)
# It then uses an engineered prompt to force GPT to think "more carefully" before answering
# We query GPT n times (default == 3) and collect the answers
# We then use GPT to compare and contrast to finally come up with a final answer
# We're using streamlit to make it a web app
import os
from apikey import apikey

import streamlit as st
import openai
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
os.environ['OPENAI_API_KEY'] = apikey

# Turn on this flag to see the debug output
_DEBUG_MODE = False

# Constants for all prompts used in the app
_TITLE = "Smarter GPT"
_DESCRIPTION = "A smarter GPT query, leveraging on engineered prompts to encourage GPT to think more carefully before answering."
_WELCOME_MSG = "What's on your mind today?"
_INITIAL_Q_PROMPT = "Answer the following carefully. Reflect on your answer \n"
_DOUBLE_CHECKER_PROMPT = "You are given a question and an answer below. The answer may be wrong so double check the following question and answer:"
_REPETITIONS = 2
_COMPARISON_PROMPT = "You are a researcher investigating the {repetition} answers to the question [[{question}]], each answer delimited by '''. \
    Compare and contrast these answers. Let's think about this step by step to make sure we find all inconsistencies."
_FINAL_ANSWER_PROMPT = "You are a resolver tasked to 1) find the best answer based on a compare-contrast opinion below, delimited by <<< and >>> \
    2) Improve on the answer. Let's think about this step by step to make sure we have the correct answer."
_SUMMARY_PROMPT = "Summarize the following answer to be more concise and straight to the point. Remove any unnecessary explanation: "



def initialize():
    # GUI using streamlit
    st.title(_TITLE)
    st.write(_DESCRIPTION)
    question = st.text_input(_WELCOME_MSG)
    model_names = ['gpt-4', 'gpt-3.5-turbo']
    model = st.selectbox('Select Model', model_names)
    submit_button = st.button('Submit')
    

    # set of answers
    initial_answers = []
    checked_answer = []

    # Prompt Templates
    # initial question
    initial_q_template = PromptTemplate(
        input_variables=['question', 'initial_q_prompt'],
        template="{initial_q_prompt} Question: {question}"
    )

    # double checker of answer to initial question
    double_checker_template = PromptTemplate(
        input_variables=['double_checker_prompt','question', 'answer'],
        template="{double_checker_prompt} Question: {question} Answer: {answer}"
    )

    # comparison of collected answers
    comparison_template = PromptTemplate(
        input_variables=['answers', 'comparison_prompt'],
        template="{comparison_prompt} {answers}"
    )

    # synthesis of answer comparison
    final_answer_template = PromptTemplate(
        input_variables=['comparison', 'final_answer_prompt'],
        template="{final_answer_prompt} <<<{comparison}>>>"
    )

    # summary of final answer to be more concise
    summary_template = PromptTemplate(
        input_variables=['summary_prompt', 'final_answer'],
        template="{summary_prompt} {final_answer}"
    )

    # api key retrieval
    openai.api_key = apikey
    # llms initialization
    llms = ChatOpenAI(temperature=0.9, client=None, model=model, max_tokens= 500 )

    # chain initialization
    initial_q_chain = LLMChain(llm=llms, prompt=initial_q_template, verbose=True, output_key='answer', )
    double_checker_chain = LLMChain(llm=llms, prompt=double_checker_template, verbose=True, output_key='checked_answer', )
    comparison_chain = LLMChain(llm=llms, prompt=comparison_template, verbose=True, output_key='comparison')
    final_answer_chain = LLMChain(llm=llms, prompt=final_answer_template, verbose=True, output_key='final_answer')
    summary_chain = LLMChain(llm=llms, prompt=summary_template, verbose=True, output_key='summary')

    # chain for initial question and double-checking of answer
    question_chain = SequentialChain(chains=[initial_q_chain, double_checker_chain],
                                        input_variables=['question', 'initial_q_prompt', 'double_checker_prompt'],
                                        output_variables=['answer', 'checked_answer'],
                                        verbose=True
                                        )
    # chain for comparison of answers and synthesis of final answer, as well as summarization
    sequential_chain = SequentialChain(chains=[comparison_chain, final_answer_chain, summary_chain], 
                                       input_variables=['answers','comparison_prompt', 'final_answer_prompt', 'summary_prompt'], 
                                       output_variables=['comparison', 'final_answer', 'summary'],
                                       verbose=True
                                       )


    # if submit button is pressed and question is not empty, initiate chains
    if submit_button and question != '':

        # Example Questions: How long will it take to reach the sun if I'm travelling at the speed of 1 million kms per hour?
        #                    I left 5 clothes to dry out in the sun. It took them 5 hours to dry completely. How long would it take to dry 30 clothes?

        # loading icon
        with st.spinner('Thinking...'):

            # we ask GPT to answer the question and double_check the answer for _REPETITIONS times
            # we then collect the answers and the double-checked results for comparison
            for x in range(_REPETITIONS):
                output = question_chain({'question':question, \
                                        'initial_q_prompt':_INITIAL_Q_PROMPT, 'double_checker_prompt': _DOUBLE_CHECKER_PROMPT})
                initial_answers.append(output['answer'])
                initial_answers.append(output['checked_answer'])
                checked_answer.append(output['checked_answer'])
                

            # concatenate initial_answers, with each answer enclosed in '''
            answers = "'''\n'''".join(initial_answers)
            # enclose answers with ''' delimeter
            answers = "'''"+answers+"'''"


            # we ask GPT to compare the answers and synthesize the final answer
            formatted_comparison_prompt = _COMPARISON_PROMPT.format(repetition=_REPETITIONS, question=question)
            final_answer = sequential_chain({'answers':answers, 'comparison_prompt':formatted_comparison_prompt, \
                                             'final_answer_prompt': _FINAL_ANSWER_PROMPT, 'summary_prompt': _SUMMARY_PROMPT})

            
        # debug output
        if _DEBUG_MODE:
            st.write("================================== INITIAL ANSWERS ==================================")
            st.write(answers)

            checked_answers = "'''\n'''".join(checked_answer)
            st.write("================================== CHECKED ANSWERS ==================================")
            st.write(checked_answers)

            st.write("================================== ASWER ==================================")

        # output final summarized answer
        st.write(final_answer['summary'])



def main():
    initialize()
    

if __name__ == "__main__":
    main()