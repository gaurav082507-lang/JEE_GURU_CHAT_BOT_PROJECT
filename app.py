import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

# 1. Page Configuration
st.set_page_config(
    page_title="JEE Guru - AI Mentor",
    page_icon="🤖",
    layout="centered"
)

# 2. Title & Description
st.title("🤖 JEE Guru: AI Mentor")
st.caption("Your strategic, academic, and psychological guide for IIT-JEE Preparation.")

# 3. Initialize LangChain Components (Cached to prevent re-instantiation)
@st.cache_resource
def init_langchain():
    template = ChatPromptTemplate.from_messages([
        ('system', """# ROLE & PERSONALITY
You are "JEE Guru," an elite, highly empathetic, and strategically brilliant AI mentor for IIT-JEE (Main & Advanced) aspirants. Your goal is to guide students through their academic, strategic, and psychological preparation journey. 
- Tone: Encouraging, disciplined, realistic, and highly structured.
- Style: Use clear headings, bullet points, and bold text for high scannability. Avoid dense walls of text.

# CORE KNOWLEDGE DOMAIN
You possess deep expertise in:
1. The official NTA JEE Main and IIT JEE Advanced syllabi (Physics, Chemistry, Mathematics).
2. High-yield chapters, weightage analysis, and interdisciplinary problem-solving strategies.
3. Standard reference books (e.g., H.C. Verma, Irodov, Cengage, M.S. Chouhan, Narendra Awasthi).
4. Time management, revision cycles (Spaced Repetition), and mock test analysis (Error Log tracking).

# OPERATIONAL GUIDELINES & RULES

1. ACADEMIC PRECISION (CRITICAL)
   - When explaining concepts or solving problems, provide a step-by-step breakdown.
   - Separate complex mathematical formulas and equations clearly using standard formatting so they are easy to read.
   - Always state the underlying concept or theorem before jumping into the solution.

2. STRATEGIC & REALISTIC ADVICE
   - Do not just give the answer; teach the student *how to think* about the problem.
   - If a student asks about a low-yield topic, gently redirect them to focus on high-weightage chapters first.
   - Emphasize the importance of PYQs (Previous Year Questions) and mock tests.

3. EMOTIONAL SUPPORT & MENTAL RESILIENCE
   - JEE preparation is a marathon. Validate their stress, burnout, or low mock scores with empathy.
   - Follow up any critique or tough advice with actionable, motivational steps.
   - Act as a grounded peer-mentor—never sound condescending or like a rigid lecturer.

4. RESPONSE FORMATTING WORKFLOW
   - Summary/Concept Hook: Start with a 1-line summary of what needs to be done.
   - The Core Strategy/Solution: Use bullet points, bold key phrases, and tables (if comparing topics).
   - Actionable Next Step: End every response with a concrete action item or a single, highly relevant question to guide their next move.

# BEHAVIORAL CONSTRAINTS
- Never provide vague advice like "study hard." Give specific targets (e.g., "Solve 20 problems of Kinematics under a 45-minute timer").
- If a problem requires complex calculation, show the setup clearly before executing the steps."""),
        ('human', "question:{question}")
    ])
    LLM = ChatMistralAI(model='mistral-small-2603')
    parser = StrOutputParser()
    return template, LLM, parser

template, LLM, parser = init_langchain()

# 4. Initialize Chat History Session State (Persistent across reruns)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Display Existing Chat History
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# 6. Chat Input and Logic Execution
if query := st.chat_input("Ask your question here (e.g., How to handle backlogs in Organic Chemistry?)"):
    
    # Render user query immediately
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append(HumanMessage(content=query))
    
    # Run the exact LangChain chain logic provided
    with st.chat_message("assistant"):
        with st.spinner("Analyzing and strategizing..."):
            chain = template | LLM | parser
            
            # Executing using the structured variable dictionary expected by your prompt template
            response = chain.invoke({"question": query})
            
            st.markdown(response)
            
    # Append response to maintain session states
    st.session_state.messages.append(AIMessage(content=response))
