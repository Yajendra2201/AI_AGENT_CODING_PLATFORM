import streamlit as st
import pandas as pd
import numpy as np
import time
import traceback
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title='AI Agent Arena', page_icon='🤖', layout='wide')

DOCUMENTS = [
    {'id':'leave_policy','title':'Leave Policy','text':'Employees receive 20 paid leaves per year. Up to 10 unused paid leaves can be carried forward to the next year. Leave requests are subject to manager approval and business requirements.'},
    {'id':'remote_work_policy','title':'Remote Work Policy','text':'Employees may work remotely for up to 3 days per week with manager approval. Remote work is subject to team requirements and company guidelines.'},
    {'id':'travel_policy','title':'Travel Policy','text':'For domestic business travel, hotel reimbursement is capped at INR 5000 per night. Employees must submit valid receipts and follow the travel approval process.'},
    {'id':'insurance_policy','title':'Health Insurance Policy','text':'Employees are covered under the company health insurance plan up to INR 500000 per year. Coverage is subject to the terms, exclusions and conditions of the insurance policy.'},
    {'id':'expense_policy','title':'Expense Policy','text':'Business expenses must be submitted with valid receipts. Expense claims should normally be submitted within 30 days of the expense.'},
    {'id':'bonus_policy','title':'Performance Bonus Policy','text':'Performance bonuses are based on company and individual performance. A positive performance review does not guarantee a bonus.'},
]

CHALLENGES = {
    "HR Policy Assistant": {
        "description": "Answer employee questions using the company policy knowledge base.",
        "difficulty": "Beginner",
        "tests": [
            {"id": 1, "query": "How many paid leaves do employees receive per year?", "expected_source": "leave_policy", "expected_facts": ["20", "paid leave"], "reference": "Employees receive 20 paid leaves per year."},
            {"id": 2, "query": "How many unused paid leaves can be carried forward?", "expected_source": "leave_policy", "expected_facts": ["10", "carried forward"], "reference": "Up to 10 unused paid leaves can be carried forward to the next year."},
            {"id": 3, "query": "How many days can employees work remotely per week?", "expected_source": "remote_work_policy", "expected_facts": ["3", "remote", "manager approval"], "reference": "Employees may work remotely for up to 3 days per week with manager approval."},
            {"id": 4, "query": "What is the domestic hotel reimbursement limit?", "expected_source": "travel_policy", "expected_facts": ["5000", "night"], "reference": "Domestic business travel hotel reimbursement is capped at INR 5000 per night."},
            {"id": 5, "query": "What is the annual health insurance coverage?", "expected_source": "insurance_policy", "expected_facts": ["500000", "year"], "reference": "The company health insurance plan covers employees up to INR 500000 per year."},
            {"id": 6, "query": "Does a positive performance review guarantee a bonus?", "expected_source": "bonus_policy", "expected_facts": ["not", "guarantee", "bonus"], "reference": "No. A positive performance review does not guarantee a bonus."},
        ],
    },
    "Travel Policy Agent": {
        "description": "Answer business travel and reimbursement questions.",
        "difficulty": "Beginner",
        "tests": [
            {"id": 1, "query": "What is the domestic hotel reimbursement cap?", "expected_source": "travel_policy", "expected_facts": ["5000", "night"], "reference": "Domestic hotel reimbursement is capped at INR 5000 per night."},
            {"id": 2, "query": "What is required for hotel reimbursement?", "expected_source": "travel_policy", "expected_facts": ["receipts"], "reference": "Employees must submit valid receipts."},
            {"id": 3, "query": "Do business trips need approval?", "expected_source": "travel_policy", "expected_facts": ["approval"], "reference": "Employees must follow the travel approval process."},
        ],
    },
    "Remote Work Agent": {
        "description": "Answer remote-work eligibility and policy questions.",
        "difficulty": "Beginner",
        "tests": [
            {"id": 1, "query": "How many remote days are allowed each week?", "expected_source": "remote_work_policy", "expected_facts": ["3", "remote"], "reference": "Employees may work remotely for up to 3 days per week."},
            {"id": 2, "query": "Is manager approval required for remote work?", "expected_source": "remote_work_policy", "expected_facts": ["manager approval"], "reference": "Remote work requires manager approval."},
            {"id": 3, "query": "What else can affect remote work?", "expected_source": "remote_work_policy", "expected_facts": ["team requirements"], "reference": "Remote work is subject to team requirements."},
        ],
    },
    "Insurance Assistant": {
        "description": "Answer employee health insurance questions.",
        "difficulty": "Intermediate",
        "tests": [
            {"id": 1, "query": "What is the annual health insurance limit?", "expected_source": "insurance_policy", "expected_facts": ["500000", "year"], "reference": "Coverage is up to INR 500000 per year."},
            {"id": 2, "query": "Is insurance coverage unconditional?", "expected_source": "insurance_policy", "expected_facts": ["terms", "exclusions"], "reference": "Coverage is subject to policy terms and exclusions."},
            {"id": 3, "query": "What kind of insurance does the company provide?", "expected_source": "insurance_policy", "expected_facts": ["health insurance"], "reference": "Employees are covered under the company health insurance plan."},
        ],
    },
    "Expense Management Agent": {
        "description": "Answer questions about employee expense claims.",
        "difficulty": "Intermediate",
        "tests": [
            {"id": 1, "query": "What must I submit with an expense claim?", "expected_source": "expense_policy", "expected_facts": ["receipts"], "reference": "Expense claims must include valid receipts."},
            {"id": 2, "query": "When should expenses normally be submitted?", "expected_source": "expense_policy", "expected_facts": ["30 days"], "reference": "Expense claims should normally be submitted within 30 days."},
            {"id": 3, "query": "Are receipts required for business expenses?", "expected_source": "expense_policy", "expected_facts": ["receipts"], "reference": "Business expenses must be submitted with valid receipts."},
        ],
    },
    "Leave Management Agent": {
        "description": "Answer leave entitlement and carry-forward questions.",
        "difficulty": "Beginner",
        "tests": [
            {"id": 1, "query": "How many paid leaves do employees get?", "expected_source": "leave_policy", "expected_facts": ["20"], "reference": "Employees receive 20 paid leaves per year."},
            {"id": 2, "query": "How many leaves can be carried forward?", "expected_source": "leave_policy", "expected_facts": ["10", "carried forward"], "reference": "Up to 10 unused paid leaves can be carried forward."},
            {"id": 3, "query": "Who approves leave requests?", "expected_source": "leave_policy", "expected_facts": ["manager approval"], "reference": "Leave requests are subject to manager approval."},
        ],
    },
    "Performance Bonus Agent": {
        "description": "Explain the company performance bonus policy.",
        "difficulty": "Intermediate",
        "tests": [
            {"id": 1, "query": "Is a bonus guaranteed after a positive review?", "expected_source": "bonus_policy", "expected_facts": ["not", "guarantee"], "reference": "A positive performance review does not guarantee a bonus."},
            {"id": 2, "query": "What determines performance bonuses?", "expected_source": "bonus_policy", "expected_facts": ["company", "individual performance"], "reference": "Performance bonuses are based on company and individual performance."},
            {"id": 3, "query": "Does individual performance matter for bonuses?", "expected_source": "bonus_policy", "expected_facts": ["individual performance"], "reference": "Individual performance is one factor in performance bonuses."},
        ],
    },
    "Policy Router Agent": {
        "description": "Identify the correct policy area before answering the question.",
        "difficulty": "Advanced",
        "tests": [
            {"id": 1, "query": "Can I work from home three days a week?", "expected_source": "remote_work_policy", "expected_facts": ["3", "remote"], "reference": "Remote work is allowed up to 3 days per week."},
            {"id": 2, "query": "How much can my hotel cost during domestic travel?", "expected_source": "travel_policy", "expected_facts": ["5000"], "reference": "Domestic hotel reimbursement is capped at INR 5000 per night."},
            {"id": 3, "query": "How many unused leaves can I carry forward?", "expected_source": "leave_policy", "expected_facts": ["10"], "reference": "Up to 10 unused paid leaves can be carried forward."},
        ],
    },
    "Multi-Policy Assistant": {
        "description": "Retrieve and combine information across different company policies.",
        "difficulty": "Advanced",
        "tests": [
            {"id": 1, "query": "Compare remote work and leave approval requirements.", "expected_source": "remote_work_policy", "expected_facts": ["manager approval"], "reference": "Both remote work and leave requests involve manager approval."},
            {"id": 2, "query": "What is the hotel cap and what documentation is needed for expenses?", "expected_source": "travel_policy", "expected_facts": ["5000", "receipts"], "reference": "Hotel reimbursement is capped at INR 5000 per night and expenses require valid receipts."},
            {"id": 3, "query": "What are the insurance limit and expense submission deadline?", "expected_source": "insurance_policy", "expected_facts": ["500000", "30 days"], "reference": "Insurance coverage is up to INR 500000 per year and expenses should normally be submitted within 30 days."},
        ],
    },
    "Grounded Answer Agent": {
        "description": "Prioritize factual, grounded answers and avoid unsupported claims.",
        "difficulty": "Advanced",
        "tests": [
            {"id": 1, "query": "Does the company guarantee a performance bonus?", "expected_source": "bonus_policy", "expected_facts": ["not", "guarantee"], "reference": "The company does not guarantee a performance bonus."},
            {"id": 2, "query": "What is the health insurance coverage?", "expected_source": "insurance_policy", "expected_facts": ["500000"], "reference": "Health insurance coverage is up to INR 500000 per year."},
            {"id": 3, "query": "What is the remote work allowance?", "expected_source": "remote_work_policy", "expected_facts": ["3", "manager approval"], "reference": "Remote work is allowed up to 3 days per week with manager approval."},
        ],
    },
}

DEFAULT_AGENT_CODE = '''class MyAgent:
    def run(self, query):
        docs = search_knowledge_base(query, top_k=3)
        context_parts = []
        for d in docs:
            context_parts.append("DOCUMENT: " + d["title"] + "\\n" + d["text"])
        context = "\\n\\n".join(context_parts)
        prompt = (
            "Answer the employee question using ONLY the policy context below. "
            "Do not invent information. If the answer is not present, say it is not available.\\n\\n"
            "POLICY CONTEXT:\\n" + context + "\\n\\nQUESTION:\\n" + query
        )
        return llm_generate(prompt)
'''

@st.cache_resource
def build_retriever():
    corpus = [d['title'] + ' ' + d['text'] for d in DOCUMENTS]
    v = TfidfVectorizer(stop_words='english')
    m = v.fit_transform(corpus)
    return v, m

VECTORIZER, DOC_MATRIX = build_retriever()

def search_knowledge_base(query, top_k=3):
    q = VECTORIZER.transform([query])
    sims = cosine_similarity(q, DOC_MATRIX)[0]
    idx = np.argsort(sims)[::-1][:top_k]
    return [{**DOCUMENTS[i], 'similarity':float(sims[i])} for i in idx]

@st.cache_resource(show_spinner=False)
def load_local_llm():
    """Try to load Qwen. Return an error instead of crashing the app."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=False,
        )
        return tokenizer, model, None

    except Exception as e:
        return None, None, str(e)


def extractive_fallback(prompt):
    """
    Offline fallback used when Hugging Face/model access is unavailable.
    Selects the most relevant sentence(s) from the policy context.
    """
    try:
        question_match = re.search(
            r"QUESTION:\s*(.*)",
            prompt,
            flags=re.IGNORECASE | re.DOTALL,
        )
        question = question_match.group(1).strip() if question_match else prompt

        context_match = re.search(
            r"POLICY CONTEXT:\s*(.*?)\s*QUESTION:",
            prompt,
            flags=re.IGNORECASE | re.DOTALL,
        )
        context = context_match.group(1).strip() if context_match else prompt

        blocks = [
            b.strip()
            for b in re.split(r"\n\s*\n", context)
            if b.strip()
        ]

        if not blocks:
            return "The answer is not available in the provided policy context."

        vec = TfidfVectorizer(stop_words="english")
        matrix = vec.fit_transform(blocks + [question])
        similarities = cosine_similarity(matrix[-1], matrix[:-1])[0]
        best_block = blocks[int(np.argmax(similarities))]

        best_block = re.sub(
            r"^DOCUMENT:\s*[^\n]+\n?",
            "",
            best_block,
            flags=re.IGNORECASE,
        ).strip()

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", best_block)
            if s.strip()
        ]

        if not sentences:
            return best_block

        q_words = set(re.findall(r"[a-zA-Z0-9]+", question.lower()))
        scored = []

        for sentence in sentences:
            s_words = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
            scored.append((len(q_words & s_words), sentence))

        scored.sort(reverse=True)

        selected = [s for score, s in scored[:2] if score > 0]
        return " ".join(selected) if selected else sentences[0]

    except Exception:
        return "The answer is not available in the provided policy context."


def llm_generate(prompt, max_new_tokens=180):
    tokenizer, model, load_error = load_local_llm()

    # Keep the POC usable even if Hugging Face access is unavailable.
    if tokenizer is None or model is None:
        return extractive_fallback(prompt)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful company policy assistant. "
                "Answer only from the provided policy context."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
    ).to(model.device)

    import torch

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


def build_agent_from_submission(code):
    namespace={'search_knowledge_base':search_knowledge_base,'llm_generate':llm_generate}
    exec(code, namespace)
    if 'MyAgent' not in namespace: raise ValueError('Your submission must define a class named MyAgent.')
    agent=namespace['MyAgent']()
    if not hasattr(agent,'run'): raise ValueError('MyAgent must contain a run(self, query) method.')
    return agent

def evaluate_test(agent, test):
    start=time.perf_counter()
    try:
        retrieved=search_knowledge_base(test['query'],3)
        source_ok=test['expected_source'] in [d['id'] for d in retrieved]
        answer=str(agent.run(test['query']))
        lower=answer.lower()
        missing=[f for f in test['expected_facts'] if f.lower() not in lower]
        coverage=(len(test['expected_facts'])-len(missing))/len(test['expected_facts'])*100
        expected_doc=next((d for d in retrieved if d['id']==test['expected_source']),None)
        similarity=expected_doc['similarity'] if expected_doc else 0
        retrieval=(70 if source_ok else 0)+min(max(similarity,0),1)*30
        ref_vec=VECTORIZER.transform([test['reference']]); ans_vec=VECTORIZER.transform([answer])
        generation=max(0,min(100,float(cosine_similarity(ref_vec,ans_vec)[0][0])*100))
        latency=time.perf_counter()-start
        return {'test_id':test['id'],'passed':source_ok and not missing,'answer':answer,'retrieval_score':round(retrieval,2),'generation_score':round(generation,2),'fact_coverage':round(coverage,2),'latency_sec':round(latency,3),'missing_facts':missing,'retrieved_docs':retrieved,'error':None}
    except Exception as e:
        return {'test_id':test['id'],'passed':False,'answer':'','retrieval_score':0,'generation_score':0,'fact_coverage':0,'latency_sec':round(time.perf_counter()-start,3),'missing_facts':[],'retrieved_docs':[],'error':f'{type(e).__name__}: {e}'}

def evaluate_submission(code, challenge):
    agent=build_agent_from_submission(code)
    results=[]
    progress=st.progress(0); status=st.empty()
    for i,test in enumerate(challenge['tests'],1):
        status.info(f'Running test case {i}/{len(challenge["tests"])}...')
        results.append(evaluate_test(agent,test)); progress.progress(i/len(challenge['tests']))
    status.success('Evaluation completed.'); progress.empty()
    df=pd.DataFrame(results)
    functional=float(df['passed'].mean()*100); retrieval=float(df['retrieval_score'].mean()); generation=float(df['generation_score'].mean()); coverage=float(df['fact_coverage'].mean()); avg=float(df['latency_sec'].mean())
    efficiency=max(0,min(100,100-avg*10))
    final=functional*.40+retrieval*.20+generation*.20+coverage*.15+efficiency*.05
    return results, {'functional':round(functional,2),'retrieval':round(retrieval,2),'generation':round(generation,2),'coverage':round(coverage,2),'efficiency':round(efficiency,2),'final_score':round(final,2),'avg_latency':round(avg,3)}

if 'history' not in st.session_state: st.session_state.history=[]
if 'agent_code' not in st.session_state: st.session_state.agent_code=DEFAULT_AGENT_CODE

with st.sidebar:
    st.markdown('## 🤖 AI Agent Arena')
    st.caption('Build • Test • Evaluate • Compete')
    page=st.radio('Navigation',['🏠 Home','🎯 Challenge','📜 Submissions','🏆 Leaderboard'])
    st.divider(); st.caption('Local Qwen • TF-IDF Retrieval • Deterministic Tests • Evaluation Metrics')

if page=='🏠 Home':
    st.title('🤖 AI Agent Arena')
    st.subheader('Build agents. Test them. Evaluate them. Compete.')
    st.markdown('**Core loop:** `Challenge → Agent Code → Test Cases → Evals → Score → Leaderboard`')
    a,b,c,d=st.columns(4); a.metric('Challenges',len(CHALLENGES)); b.metric('Test Cases',6); c.metric('Metrics',5); d.metric('LLM','Qwen 1.5B')
    st.divider(); st.markdown('### Evaluation dimensions')
    cols=st.columns(5)
    for col,(n,v) in zip(cols,[('Functional','PASS / FAIL'),('Retrieval','Source quality'),('Generation','Semantic quality'),('Coverage','Required facts'),('Efficiency','Latency')]):
        col.info(f'**{n}**\n\n{v}')
    st.divider(); st.markdown('### Production evolution')
    st.write('Replace the local notebook-style execution with a web frontend, FastAPI, secure Docker/microVM sandbox, PostgreSQL, Redis job queues, hidden tests, agent tracing, authentication and scalable workers.')

elif page=='🎯 Challenge':
    st.title('🎯 AI Agent Challenge')
    name=st.selectbox('Select Challenge',list(CHALLENGES.keys())); challenge=CHALLENGES[name]
    st.markdown(f'### {name}'); st.write(challenge['description'])
    x,y=st.columns(2); x.success(f'Difficulty: {challenge["difficulty"]}'); y.info(f'Test cases: {len(challenge["tests"])}')
    with st.expander('📖 Challenge Instructions',expanded=True):
        st.markdown('''Your agent must implement:\n\n```python\nclass MyAgent:\n    def run(self, query):\n        ...\n```\n\nAvailable functions:\n\n```python\nsearch_knowledge_base(query, top_k=3)\nllm_generate(prompt)\n```\n\nThe agent should retrieve relevant policy documents and generate an answer grounded in those documents.''')
    st.markdown('### 💻 Write Your Agent')
    code=st.text_area('Agent code',height=430,label_visibility='collapsed',key='agent_editor')
    c1,c2=st.columns([1,1])
    run=c1.button('▶ Run Evaluation',type='primary',use_container_width=True)
    if c2.button('↩ Reset Code',use_container_width=True): st.session_state.agent_code=DEFAULT_AGENT_CODE; st.rerun()
    if run:
        try:
            with st.spinner('Building agent and running evaluation...'):
                results,metrics=evaluate_submission(code,challenge)
            st.session_state.last_results=results; st.session_state.last_metrics=metrics
            st.session_state.history.append({'challenge':name,'score':metrics['final_score'],'passed':sum(r['passed'] for r in results),'total':len(results),'timestamp':time.strftime('%Y-%m-%d %H:%M:%S')})
        except Exception as e:
            st.error(f'Submission failed: {type(e).__name__}: {e}')
            with st.expander('Technical traceback'): st.code(traceback.format_exc())
    if 'last_metrics' in st.session_state:
        m=st.session_state.last_metrics; results=st.session_state.last_results
        st.divider(); st.markdown('## 📊 Evaluation Result')
        cs=st.columns(6)
        for col,label,key in zip(cs,['Final Score','Functional','Retrieval','Generation','Coverage','Efficiency'],['final_score','functional','retrieval','generation','coverage','efficiency']): col.metric(label,m[key] if key=='final_score' else m[key])
        st.progress(m['final_score']/100)
        st.markdown('### Test Results')
        for r in results:
            test=next(t for t in challenge['tests'] if t['id']==r['test_id'])
            with st.expander(f"{'✅ PASS' if r['passed'] else '❌ FAIL'} — Test {r['test_id']}: {test['query']}"):
                if r['error']: st.error(r['error']); continue
                st.markdown('**Agent Answer**'); st.info(r['answer'])
                a,b,c,d=st.columns(4); a.metric('Retrieval',r['retrieval_score']); b.metric('Generation',r['generation_score']); c.metric('Fact Coverage',r['fact_coverage']); d.metric('Latency',f"{r['latency_sec']}s")
                if r['missing_facts']: st.warning('Missing required facts: '+', '.join(r['missing_facts']))
                st.markdown('**Retrieved Documents**')
                for doc in r['retrieved_docs']: st.write(f"- `{doc['id']}` — similarity={doc['similarity']:.3f}")

elif page=='📜 Submissions':
    st.title('📜 Submission History')
    if not st.session_state.history: st.info('No submissions yet. Go to Challenge and run your agent.')
    else:
        st.dataframe(pd.DataFrame(st.session_state.history),use_container_width=True,hide_index=True)

else:
    st.title('🏆 Leaderboard')
    rows=[{'Rank':1,'User':'DemoUser_A','Challenge':'HR Policy Assistant','Score':92.4,'Pass Rate':'100%'},{'Rank':2,'User':'DemoUser_B','Challenge':'HR Policy Assistant','Score':88.1,'Pass Rate':'83%'},{'Rank':3,'User':'DemoUser_C','Challenge':'HR Policy Assistant','Score':85.7,'Pass Rate':'83%'}]
    if st.session_state.history:
        best=max(st.session_state.history,key=lambda x:x['score']); rows.append({'Rank':'-','User':'You','Challenge':best['challenge'],'Score':best['score'],'Pass Rate':f"{best['passed']/best['total']*100:.0f}%"})
    df=pd.DataFrame(rows).sort_values('Score',ascending=False).reset_index(drop=True); df['Rank']=range(1,len(df)+1)
    st.dataframe(df,use_container_width=True,hide_index=True)
    st.caption('Demo leaderboard for the POC. Production version would persist rankings in PostgreSQL.')

st.divider()
st.caption('⚠️ Local POC only: submitted Python code is executed with exec(). Do not expose this implementation to untrusted users; production requires isolated sandbox execution.')
