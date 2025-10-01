import streamlit as st
import requests
import base64

# ------------------- Page Config -------------------
st.set_page_config(page_title="🌊 FloatChat", layout="wide")

# ------------------- Custom CSS -------------------
st.markdown(
    """
    <style>
    /* Background ocean image */
    .stApp {
        background: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80") no-repeat center center fixed;
        background-size: cover;
        color: white;
    }

    /* Transparent dark overlay for readability */
    .overlay {
        background: rgba(0, 0, 0, 0.4);
        padding: 10px;
        border-radius: 12px;
        display: inline-block;
    }

    /* Chat bubbles */
    .user-bubble {
        background: #0288d1;
        color: white;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 10px 0;
        max-width: 70%;
        align-self: flex-end;
        font-size: 15px;
    }
    .bot-bubble {
        background: rgba(0,0,0,0.6);
        color: white;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 10px 0;
        max-width: 70%;
        align-self: flex-start;
        font-size: 15px;
    }

    /* Chat container */
    .chat-container {
        display: flex;
        flex-direction: column;
        margin: 12px 0;
    }

    /* Floating input box */
    .stTextInput>div>div>input {
        background-color: rgba(255,255,255,0.15);
        color: white;
        border-radius: 20px;
        border: 1px solid #81d4fa;
    }

    /* Top navigation bar */
    .topbar {
        width: 100%;
        padding: 15px 40px;
        background: rgba(0,0,0,0.5);
        position: fixed;
        top: 0;
        left: 0;
        z-index: 100;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .topbar a {
        margin: 0 15px;
        color: #e0f7fa;
        text-decoration: none;
        font-weight: 500;
    }
    .topbar a:hover {
        color: #80deea;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------- Top Navigation -------------------
st.markdown(
    """
    <div class="topbar">
        <div><b>🌊 FloatChat</b></div>
        <div>
            <a href="#">Home</a>
            <a href="#">Learn</a>
            <a href="#">Research</a>
            <a href="#">Community</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br><br><br>", unsafe_allow_html=True)  # spacing below navbar

# ------------------- Session State -------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

api_url = "http://127.0.0.1:8000/chat"

# ------------------- User Input -------------------
user_input = st.text_input("Type your message here...", "")

if user_input:
    with st.spinner("🤿 Thinking..."):
        try:
            res = requests.post(api_url, json={"question": user_input}).json()
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            if res["type"] == "text":
                st.session_state.chat_history.append({"role": "bot", "content": res["answer"]})

            elif res["type"] == "visualization":
                img_data = base64.b64decode(res["image"])
                st.session_state.chat_history.append(
                    {"role": "bot", "content": "📊 Here’s your visualization:", "image": img_data, "code": res["code"]}
                )

            elif res["type"] == "error":
                st.session_state.chat_history.append({"role": "bot", "content": f"⚠️ Error: {res['error']}"})

        except Exception as e:
            st.session_state.chat_history.append({"role": "bot", "content": f"❌ Request failed: {e}"})

# ------------------- Display Chat -------------------
st.markdown("### 📜 Conversation")
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-container'><div class='user-bubble'>{msg['content']}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-container'><div class='bot-bubble'>{msg['content']}</div></div>", unsafe_allow_html=True)
        if "image" in msg:
            st.image(msg["image"], use_column_width=True)
            with st.expander("🔍 Show Generated Code"):
                st.code(msg["code"], language="python")
