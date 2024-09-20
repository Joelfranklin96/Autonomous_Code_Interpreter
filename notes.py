from dotenv import load_dotenv
from langchain import hub
from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
# The above package gives the LLM the ability to write and execute python code in the interpreter.
# REPL = Read/Evaluate/Print Loop

load_dotenv()

def main():
    print("Start...")

    instructions = """You are an agent designed to write and execute python code to answer questions.
        You have access to a python REPL, which you can use to execute python code.
        If you get an error, debug your code and try again.
        Only use the output of your code to answer the question. 
        You might know the answer without running any code, but you should still run the code to get the answer.
        If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
        """
    # Here 'agent' does not refer to LangChain agent. It is meant in the english literal sense and refers to
    # the role of LLM.

    base_prompt = hub.pull("langchain-ai/react-agent-template")

    # The react-agent-template is as given below
    # ------------------------------------------------------------------------------------------------------------------
    # {instructions}

    # TOOLS:
    # ------

    # You have access to the following tools:

    # {tools}

    # To use a tool, please use the following format:

    # ```
    # Thought: Do I need to use a tool? Yes
    # Action: the action to take, should be one of [{tool_names}]
    # Action Input: the input to the action
    # Observation: the result of the action
    # ```

    # When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

    # ```
    # Thought: Do I need to use a tool? No
    # Final Answer: [your response here]
    # ```

    # Begin!

    # Previous conversation history:
    # {chat_history}

    # New input: {input}
    # {agent_scratchpad}

    #-------------------------------------------------------------------------------------------------------------------

    # Note
    # Multiple Cycles Are Allowed: Even though the prompt doesn't specifically state that the cycle can repeat
    # multiple times, the agent is designed to iterate through the Thought → Action → Observation cycle as
    # many times as necessary.

    # Default Behavior Supports Iteration: In the context of the ReAct framework and LangChain agents,
    # iteration is the default behavior.

    # If we want a single cycle, then it has to be explicitly specified.

    # As per the above prompt, the LLM comes up with a Final answer when it decides not to use a tool.
    # Until then, it uses any of the tools and the iteration continues.

    # The Final Answer does not have to be the actual solution to the initial question. The Final Answer can
    # indeed be a request for clarification or any other appropriate response to the user.

    # As per the above prompt, the LLM can provide the final answer only after deciding not to use a tool.
    # Though it is possible that the LLM uses a tool, makes an observation and obtains the final answer but
    # before providing the Final Answer, the LLM must include a thought step stating it does not need to use a tool.
    # The thought step stating it does not need to use a tool must definitely precede the step that provides the
    # final answer (As per the above prompt).

    # "New Input" in the prompt template refers to the latest message or question (follow-up) provided by the user that
    # the LLM needs to address.

    # {agent_scratchpad} represents the LLM's thinking steps in written format. It is where the agent records its
    # internal reasoning process, including Thoughts, Actions, and Observations, as it works towards generating a
    # final answer to the user's input.

    # The {agent_scratchpad} is typically used to record the agent's thinking steps for the current new_input only.
    # Not Cumulative History: It does not contain the history of thinking steps from all previous inputs.
    # Instead, it starts fresh with each new user input.

    # The above react-agent-template is different from the below react template (that we used in ice_breaker project,
    # linkedin_lookup_agent.py).

    # ------------------------------------------------------------------------------------------------------------------
    # Answer the following questions as best you can. You have access to the following tools:

    # {tools}

    # Use the following format:

    # Question: the input question you must answer
    # Thought: you should always think about what to do
    # Action: the action to take, should be one of [{tool_names}]
    # Action Input: the input to the action
    # Observation: the result of the action
    # ... (this Thought/Action/Action Input/Observation can repeat N times)
    # Thought: I now know the final answer
    # Final Answer: the final answer to the original input question

    # Begin!

    # Question: {input}
    # Thought:{agent_scratchpad}
    # ------------------------------------------------------------------------------------------------------------------

    # # The above react template doesn't take into account a follow-up question. The user asks 1 single question and
    # the LLM tries to answer it. If the user asks another follow-up question, the LLM treats it as a
    # separate independent question. There is no chat_history.

    prompt = base_prompt.partial(instructions=instructions)

    # Note
    # We plug in 2 kinds of variables namely 'input_variables' and 'partial_variables' into the 'base_prompt' to create
    # the 'prompt'. The 'input_variables' are passed in during invoking the chain while the 'partial_variables' are
    # predefined variables.

    tools = [PythonREPLTool()]
    # The Python REPL tool is a Python shell that can execute Python commands and Python code. The input to the
    # REPL tool should be a valid python code.

    # We didn't had to call the 'Tool' class or write tool name/description because the PythonREPL tool is already a
    # 'Tool' object that has been imported.
    agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # # agent_executor acts as the runtime environment that manages the execution of the agent. The AgentExecutor is
    # responsible for running the agent. It takes the agent you've created and executes it with the given tools and
    # inputs.
    # 'verbose=True' to see extra logs.

    agent_executor.invoke(
        input={
            "input": """generate and save in current working directory 15 QRcodes that point to 
            www.udemy.com/course/langchain, you have qrcode package installed already"""
        }
    )

    # Note
    # The base_prompt template has the following placeholders - (instructions, tools, tool_names, chat_history,
    # input, agent_scratchpad).
    # The 'instructions' is filled as a partial variable while defining the prompt_template.
    # The 'tools' is passed while defining the agent and agent_executor.
    # When we pass the tools parameter while defining the agent and the AgentExecutor, the {tool_names} placeholder in
    # the prompt template is also automatically filled by the AgentExecutor.
    # The 'input' is passed while invoking the agent.

    # How is 'chat_history' filled in the prompt_template?
    # The AgentExecutor maintains an internal log of all previous interactions between the user and the agent.
    # Each time we invoke the agent with a new input, the AgentExecutor appends this input and the agent's response to
    # the chat_history. The agent's response is nothing but the final answer generated by the LLM for a particular user
    # input query.
    # The agent_executor internally manages and populates the 'chat_history' placeholder in the prompt_template.

    # How is 'agent_scratchpad' filled in the prompt_template?
    # The agent_scratchpad is nothing but Thought-Action-Observation (TAO sequence).
    # The TAO sequence is appended to the agent_scratch after every iteration.
    # The agent_executor internally manages and populates the 'agent_scratchpad' placeholder in the prompt_template.
    # The agent_scratchpad allows the language model to see the agent's prior reasoning and decide on the
    # next steps accordingly.

    # Note
    # The LLM does not have direct access to my local machine, file system, or any hardware resources on my
    # local machine. The LLM is running on OpenAI's servers. The LLM can only generate text (i.e code in our case).
    # The LLM sends the information of the code and the tool to be run to the LangChain agent.
    # The LangChain agent has permissions to use the PythonREPLTool which has been already imported on my local machine.
    # The LangChain agent has permissions to run the PythonREPLTool on my local machine. And the PythonREPL tool,
    # since it's been imported on my local machine, has access to the file system. It runs the code and
    # brings changes in the file system.

    # Note
    # We are not seeing many LangChain agents in production these days because the LLM reasoning ability is not yet
    # developed enough for the usage of LangChain agents. Also LLMs like gpt-4 is expensive and we make the LangChain
    # applications work by efficiently making use of Prompt Engineering.

    # Production environments have to be reliable but LLMs are statistical creatures.
    # Crafting effective prompts (Prompt Engineering) can increase the reliability of LLMs and minimize unnecessary
    # token usage, thus lowering costs.

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="episode_info.csv",
        verbose=True,
    )

    csv_agent.invoke(
        input={"input": "how many columns are there in file episode_info.csv"}
    )
    csv_agent.invoke(
        input={
            "input": "print the seasons by ascending order of the number of episodes they have"
        }
    )

    # Note
    # The main reason we create a separate csv_agent using create_csv_agent is to leverage specialized tools and
    # configurations that are optimized for interacting with CSV files. While it's theoretically possible to modify the
    # base prompt and use the PythonREPLTool to achieve similar functionality, there are several advantages to
    # using a dedicated CSV agent.

    # The csv_agent is specifically designed to interact with CSV data by providing the Language Model (LLM) with
    # detailed context about the CSV file, including its schema and sample data (See note below).
    # It preloads the CSV into a pandas DataFrame (df), allowing the LLM to generate code that directly operates on
    # the data without needing to handle data loading and parsing.

    # While the PythonREPLTool allows the LLM to execute arbitrary Python code, it doesn't provide any context about
    # the data or preloaded datasets. The LLM would need to generate additional code to load and understand the
    # CSV file, which increases the cost, complexity and the potential for errors.

    # Note
    # In LangChain, the csv_agent is indeed an agent, not a tool. However, in the context of creating a hybrid agent
    # that can utilize multiple tools and agents, we can wrap the csv_agent inside a function, assign it a name and
    # description, and then pass these as parameters to the Tool class. This effectively turns the csv_agent into a
    # tool.

    # The LangChain agent is the commando. The LLM is the commando's brain. The set of tools are the soldiers.
    # Together they form the army to solve the objective of the user.

    # Note (How does the interaction between the LLM, Agent and Tools take place?)
    # 1) Role of LLM - To generate the Thought-Action-Action Input
    # Example of LLM's output
    # Thought: I need to calculate the standard deviation of the 'Temperature' column.
    # Action: Python REPL
    # Action Input: Input parameter passed to Python REPL (In this case, the input parameter is the code itself)

    # 2) Role of Agent - Parses the text generated by the LLM to identify actions. Executes the specified action using
    # the appropriate tool (e.g., PythonREPLTool). Collects the output (observation) from the tool execution.

    # Observation is fed back to the LLM by the agent. The LLM may decide to perform additional actions or
    # provide the final answer.
    # The LLM uses observations to refine its understanding and decide on next step (whether to go for Thought-Action-
    # Action Input iteration again or generate a final answer).

    # Interaction between LLM and csv_agent and how they arrive at the final answer
    # Initially the csv_agent runs pre built-in code on the input CSV data to provide the LLM with all the
    # necessary context about the CSV data.
    # The LLM interprets the user input query and the information about the input CSV data passed from the csv_agent,
    # then generates a Thought-Action-Action Input string. This Thought-Action-Action Input (the action may contain
    # some customized function code) is parsed by the csv_agent.
    # The csv_agent runs the customized code and forwards the observation (output of code run) to the LLM.
    # The LLM interprets the observation and then decides whether to go for another iteration of
    # Thought-Action-Action Input or if satisfied generates the final answer.

    # Example
    # Input user query: "Find the oldest employee in the company and provide their name and age."
    # Thought: "I need to identify the employee with the maximum age in the dataset and retrieve their name and age."
    # Action: "Python REPL" (This is the Python REPL inbuilt tool within the csv_agent).
    # Action Input: "Code to solve the objective"
    # import pandas as pd
    # df = pd.read_csv('employees.csv')
    # oldest_employee = df.loc[df['Age'].idxmax()]
    # name = oldest_employee['Name']
    # age = oldest_employee['Age']
    # print(f"The oldest employee is {name}, who is {age} years old.")
    # Observation (Output of code): "The oldest employee is John Smith, who is 65 years old."
    # The csv_agent sends the observation to the LLM.
    # The LLM is satisfied and hence generates the below final answer.
    # Final answer: "The oldest employee in the company is John Smith, who is 65 years old."


    # Note
    # The csv_agent has a built-in Python REPL tool. The built-in Python REPL tool is different from the one that
    # we imported.

    # user input query - "In episode_info.csv, which writer wrote the most episodes and how many episodes did he write?"
    # The final answer may not be correct because the agent passed information/context about the input. The information
    # such as some sample data and features (number of columns/rows/mean/std etc) of dataset.
    # The LLM might have added a filter like df=df[df['Season'] <= 5] if the sample data only included seasons up to
    # 5, as the LLM uses heuristics based on the sample data to generate code.
    # Heuristics are problem-solving methods that use practical approaches for immediate solutions but aren't
    # guaranteed to be perfect.


if __name__ == '__main__':
    main()

    # Note
    # In this 'notes.py', we created 2 agents. One is a ReAct based agent which uses the ReAct algorithm with the
    # ReAct agent template prompt. The other is csv_agent which under the hood uses the Pandas agent which in turn uses
    # the ReAct algorithm.

    # Note
    # How does the Pandas agent use the ReAct algorithm?
    # Initially the csv_agent has built-in code to fetch details about the dataframe's columns and data types, a few
    # rows of data to illustrate the content and guidelines for LLM on how to interact with df. This provides the LLM
    # the necessary context to generate appropriate code.
    # LLM Generates Thought-Action-Action Input string. This string is parsed by the csv_agent and the Action (code
    # generated by LLM) is run by the agent. The observation (output of code) is sent to the LLM.
    # The LLM then decides whether to repeat the Thought-Action-Action Input or if satisfied, it generates the final
    # answer. Hence this is the ReAct algorithm.

    # ReAct (Reasoning-Action) Algorithm
    # The reasoning part (Thought-Action-Action Input) is done by the LLM.
    # The action part (executing the action) is done by the agent.
    # The agent feeds the observation (output of action) to the LLM.
    # The LLM decides whether to repeat the cycle or if satisfied, it generates the final answer.
    # This iteration of Reasoning, Action and Feedback loop is the ReAct Algorithm.

    # Since the csv_agent (or the underneath pandas agent) leverages the ReAct (Reasoning and Acting) paradigm,
    # there is indeed a prompt template at play behind the scenes.

    # Hence the below code is wrong
    # csv_agent.invoke(
    #     input={"user_query": "how many columns are there in file episode_info.csv"}
    # )
    # Because only 'input' (not 'user_query') is a placeholder in the prompt_template.

    # 'input', 'path', 'chat_history', and 'agent_scratchpad' are the key placeholders in the prompt template for a
    # ReAct-based csv_agent. I guess there could also be 'tools' and 'tool_names'
    # placeholders (Not sure because the only tool that the csv_agent uses is the in built PythonREPL tool).

