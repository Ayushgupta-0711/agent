import streamlit as st
import pandas as pd
import sqlite3
import io
import re
from typing import Tuple, List
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import numpy as np

st.set_page_config(page_title="Multi-Agent Data + Research Assistant", layout="wide")

# ----------------------------- Utilities -----------------------------

def to_sqlite(df: pd.DataFrame, table_name: str, conn: sqlite3.Connection):
    df.to_sql(table_name, conn, if_exists='replace', index=False)


def detect_file_type(filename: str) -> str:
    if filename.lower().endswith(('.csv', '.tsv')):
        return 'csv'
    if filename.lower().endswith(('.xls', '.xlsx')):
        return 'excel'
    if filename.lower().endswith('.pdf'):
        return 'pdf'
    return 'unknown'


# ----------------------------- Data Intelligence Agent -----------------------------

def parse_data_query(query: str, df: pd.DataFrame) -> Tuple[str, dict]:
    """Heuristic parser returns an intent and parameters."""
    q = query.lower()
    intent = 'unknown'
    params = {}

    if any(w in q for w in ['plot', 'trend', 'chart', 'show a', 'visual', 'visualize']):
        intent = 'plot'
    elif any(w in q for w in ['total', 'sum', 'how much', 'what was the']) and any(w in q for w in ['sales','revenue','amount','profit']):
        intent = 'aggregate'
        params['agg'] = 'sum'
    elif any(w in q for w in ['average','mean','avg']):
        intent = 'aggregate'
        params['agg'] = 'mean'
    elif any(w in q for w in ['top', 'highest', 'rank']):
        intent = 'topn'
    elif any(w in q for w in ['count', 'how many']):
        intent = 'count'

    # detect numeric N for top N
    m = re.search(r'top\s*(\d+)', q)
    if m:
        params['n'] = int(m.group(1))

    # find likely metric column
    candidates = df.columns.astype(str).tolist()
    metric = None
    for col in candidates:
        if re.search(r'sales|revenue|amount|total|price|profit', col.lower()):
            metric = col
            break
    if metric is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        metric = numeric_cols[0] if numeric_cols else None

    params['metric'] = metric

    # group-by detection
    for g in ['product', 'category', 'region', 'country', 'month', 'quarter', 'year']:
        for col in candidates:
            if g in col.lower():
                params['groupby'] = col
                break
        if 'groupby' in params:
            break

    return intent, params


def handle_data_query(query: str, df: pd.DataFrame):
    intent, params = parse_data_query(query, df)
    metric = params.get('metric')
    groupby = params.get('groupby')

    if intent == 'aggregate':
        agg = params.get('agg','sum')
        if metric is None:
            return f"Couldn't detect a numeric metric column to {agg}.", None
        if agg == 'sum':
            total = df[metric].sum()
            return f"Total {metric}: {total}", None
        else:
            meanv = df[metric].mean()
            return f"Average {metric}: {meanv}", None

    elif intent == 'plot':
        if metric is None:
            return "Couldn't detect a numeric metric to plot.", None
        time_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'month' in col.lower() or 'year' in col.lower():
                time_col = col
                break
        if groupby and groupby in df.columns:
            if time_col:
                grouped = df.groupby([time_col, groupby])[metric].sum().reset_index()
                fig = px.line(grouped, x=time_col, y=metric, color=groupby, title=f"{metric} trend by {groupby}")
                return "", fig
            else:
                agg = df.groupby(groupby)[metric].sum().reset_index().sort_values(metric, ascending=False)
                fig = px.bar(agg, x=groupby, y=metric, title=f"{metric} by {groupby}")
                return "", fig
        else:
            if time_col:
                ts = df.groupby(time_col)[metric].sum().reset_index()
                fig = px.line(ts, x=time_col, y=metric, title=f"{metric} trend over {time_col}")
                return "", fig
            else:
                fig = px.histogram(df, x=metric, nbins=20, title=f"Distribution of {metric}")
                return "", fig

    elif intent == 'topn':
        n = params.get('n', 5)
        if metric is None:
            return f"Couldn't detect a numeric metric for top {n}.", None
        if groupby and groupby in df.columns:
            agg = df.groupby(groupby)[metric].sum().reset_index().sort_values(metric, ascending=False).head(n)
            fig = px.bar(agg, x=groupby, y=metric, title=f"Top {n} {groupby} by {metric}")
            return agg, fig
        else:
            cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
            if cat_cols:
                gb = df.groupby(cat_cols[0])[metric].sum().reset_index().sort_values(metric, ascending=False).head(n)
                fig = px.bar(gb, x=cat_cols[0], y=metric, title=f"Top {n} {cat_cols[0]} by {metric}")
                return gb, fig
            else:
                return "No categorical column found to compute top N.", None

    elif intent == 'count':
        return f"Number of rows: {len(df)}", None

    else:
        return "I'm not sure which operation to perform.", None


# ----------------------------- Research Intelligence Agent -----------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    out = []
    for p in reader.pages:
        try:
            out.append(p.extract_text() or '')
        except Exception:
            out.append('')
    return '\n'.join(out)


def summarize_text(text: str, max_sentences: int = 5) -> str:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip())>20]
    if not sentences:
        return "(No extractable text)"
    vec = TfidfVectorizer(stop_words='english')
    X = vec.fit_transform(sentences)
    scores = X.sum(axis=1).A1
    ranked_ix = np.argsort(scores)[::-1][:max_sentences]
    top_sents = [sentences[i] for i in sorted(ranked_ix)]
    return '\n'.join(top_sents)


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
    X = vec.fit_transform([text])
    feature_array = np.array(vec.get_feature_names_out())
    tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]
    top_n = feature_array[tfidf_sorting][:top_k]
    return [k for k in top_n if len(k)>2]


def research_qa(question: str, doc_text: str, top_k: int = 3) -> str:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', doc_text) if len(s.strip())>20]
    if not sentences:
        return "(No text to answer from)"
    vec = TfidfVectorizer(stop_words='english')
    S = vec.fit_transform(sentences)
    qv = vec.transform([question])
    sims = cosine_similarity(qv, S).flatten()
    top_ix = sims.argsort()[::-1][:top_k]
    answers = [sentences[i] for i in top_ix if sims[i]>0]
    if answers:
        return '\n\n'.join(answers)
    else:
        return "Couldn't find a direct excerpt."


# ----------------------------- Agent Coordinator -----------------------------

def route_query(query: str, data_df: pd.DataFrame, research_text: str):
    q = query.lower()
    data_keywords = ['sales', 'revenue', 'plot', 'chart', 'top', 'sum', 'average', 'profit', 'customers']
    research_keywords = ['paper', 'pdf', 'study', 'method', 'conclusion', 'summary', 'findings', 'literature']
    data_score = sum(1 for kw in data_keywords if kw in q)
    research_score = sum(1 for kw in research_keywords if kw in q)

    results = {}
    if data_score >= research_score and data_df is not None:
        results['data'] = handle_data_query(query, data_df)
    if research_score >= data_score and research_text:
        results['research'] = {
            'summary': summarize_text(research_text, max_sentences=5),
            'keywords': extract_keywords(research_text, top_k=10),
            'answer': research_qa(query, research_text)
        }
    if not results:
        if data_df is not None:
            results['data'] = handle_data_query(query, data_df)
        if research_text:
            results['research'] = {
                'summary': summarize_text(research_text, max_sentences=5),
                'keywords': extract_keywords(research_text, top_k=10),
                'answer': research_qa(query, research_text)
            }
    return results


# ----------------------------- Streamlit UI -----------------------------

st.title("📊📚 Multi-Agent: Data Intelligence + Research Assistant")

if 'data_dfs' not in st.session_state:
    st.session_state['data_dfs'] = {}
if 'sqlite_conn' not in st.session_state:
    st.session_state['sqlite_conn'] = sqlite3.connect(':memory:')
if 'research_docs' not in st.session_state:
    st.session_state['research_docs'] = {}

# Sidebar upload
st.sidebar.header("Ingest Files")
uploaded = st.sidebar.file_uploader("Upload CSV/Excel/PDF", accept_multiple_files=True)
if uploaded:
    for f in uploaded:
        ftype = detect_file_type(f.name)
        if ftype == 'csv':
            df = pd.read_csv(f)
            st.session_state['data_dfs'][f.name] = df
            to_sqlite(df, re.sub('[^0-9a-zA-Z]+','_', f.name), st.session_state['sqlite_conn'])
            st.sidebar.success(f"Ingested CSV: {f.name}")
        elif ftype == 'excel':
            df = pd.read_excel(f)
            st.session_state['data_dfs'][f.name] = df
            to_sqlite(df, re.sub('[^0-9a-zA-Z]+','_', f.name), st.session_state['sqlite_conn'])
            st.sidebar.success(f"Ingested Excel: {f.name}")
        elif ftype == 'pdf':
            raw = f.read()
            txt = extract_text_from_pdf(raw)
            st.session_state['research_docs'][f.name] = txt
            st.sidebar.success(f"Ingested PDF: {f.name}")
        else:
            st.sidebar.warning(f"Unsupported file type: {f.name}")

# Tabs
tab1, tab2, tab3 = st.tabs(["Data Intelligence", "Research Intelligence", "Unified Chat"])

with tab1:
    st.header("Data Intelligence Agent")
    if st.session_state['data_dfs']:
        sel = st.selectbox("Choose dataset", options=list(st.session_state['data_dfs'].keys()))
        df = st.session_state['data_dfs'][sel]
        st.subheader("Preview")
        st.dataframe(df.head(200))

        query = st.text_input("Ask a question about this dataset")
        if st.button("Run Data Query") and query:
            output, fig = handle_data_query(query, df)
            if isinstance(output, pd.DataFrame) and not output.empty:
                st.write("**Result table:**")
                st.dataframe(output)
            elif isinstance(output, str) and output:
                st.write(output)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload a CSV or Excel file in the sidebar to begin.")

with tab2:
    st.header("Research Intelligence Agent")
    if st.session_state['research_docs']:
        sel_r = st.selectbox("Choose document", options=list(st.session_state['research_docs'].keys()))
        doc_text = st.session_state['research_docs'][sel_r]
        st.subheader("Auto Summary")
        st.write(summarize_text(doc_text, max_sentences=5))
        st.subheader("Keywords")
        st.write(', '.join(extract_keywords(doc_text, top_k=15)))

        q = st.text_input("Ask a question about the document")
        if st.button("Run Research Q&A") and q:
            ans = research_qa(q, doc_text, top_k=3)
            st.subheader("Answer (excerpt from the document)")
            st.write(ans)
    else:
        st.info("Upload at least one PDF in the sidebar to analyze research documents.")

with tab3:
    st.header("Unified Chat")
    data_df = None
    if st.session_state['data_dfs']:
        data_choice = st.selectbox("Select dataset (optional)", options=['(none)']+list(st.session_state['data_dfs'].keys()))
        if data_choice != '(none)':
            data_df = st.session_state['data_dfs'][data_choice]
    research_text = None
    if st.session_state['research_docs']:
        doc_choice = st.selectbox("Select document (optional)", options=['(none)']+list(st.session_state['research_docs'].keys()))
        if doc_choice != '(none)':
            research_text = st.session_state['research_docs'][doc_choice]

    user_q = st.text_area("Enter your question")
    if st.button("Ask") and user_q:
        with st.spinner('Routing and computing...'):
            res = route_query(user_q, data_df, research_text)
        st.subheader("Results")
        if 'data' in res:
            out, fig = res['data']
            st.markdown("### Data Agent Response")
            if isinstance(out, pd.DataFrame) and not out.empty:
                st.dataframe(out)
            elif isinstance(out, str) and out:
                st.write(out)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        if 'research' in res:
            st.markdown("### Research Agent Response")
            st.write("**Summary:**")
            st.write(res['research']['summary'])
            st.write("**Keywords:**")
            st.write(', '.join(res['research']['keywords']))
            st.write("**Answer (excerpts):**")
            st.write(res['research']['answer'])

# Footer
st.markdown("---")
st.write("**Tips:** Use clear phrases like 'Plot', 'Top N', 'Total sales', or 'Summarize the paper'.")
