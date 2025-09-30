import streamlit as st
import plotly.graph_objects as go
import random
import pandas as pd
import numpy as np

# ------------- Helper functions -------------

def generate_candlestick_data(case_number: int, n: int = 200):
    """
    Generate random OHLC candlestick data with seed tied to case_number
    for reproducibility.
    """
    np.random.seed(case_number)  # reproducible per case_number

    # simulate random walk
    dt = 1/252
    mu = 0
    sigma = 0.5
    S0 = 100

    # Generate geometric Brownian motion
    t = np.linspace(0, n*dt, n)
    W = np.random.standard_normal(size=n)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu - 0.5*sigma**2) * t + sigma * W
    S = S0 * np.exp(X)

    # Build OHLC
    close = S
    open_ = np.concatenate([[S0], S[:-1]])
    high = np.maximum(open_, close) + np.random.rand(n)*1
    low = np.minimum(open_, close) - np.random.rand(n)*1
    
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=dates)
    return df

def plot_candlestick_predict(df: pd.DataFrame, case_number: int):
    """
    Show only first 80% candles, mask last 20% with NaN.
    """
    split_idx = int(len(df) * 0.8)
    
    df_predict = df.copy()
    df_predict.iloc[split_idx:] = np.nan    # hide last 20%
    vline_date = df.index[split_idx]
    last_close = df["Close"].iloc[split_idx-1]
    
    fig = go.Figure()
    # plot partial candles
    fig.add_trace(go.Candlestick(
        x=df_predict.index,
        open=df_predict["Open"],
        high=df_predict["High"],
        low=df_predict["Low"],
        close=df_predict["Close"],
        name="Predict Data"
    ))
    
    # Add vertical line
    fig.add_vline(
        x=vline_date,
        line_width=1,
        line_dash="dash",
        line_color="red"
    )

    # Add horizontal ray starting from vline_date
    fig.add_shape(
        type="line",
        x0=vline_date, 
        x1=df.index[-1],       # extend to last available date
        y0=last_close, 
        y1=last_close,
        line=dict(color="blue", dash="dot", width=1),
        xref="x", yref="y"
    )
    
    fig.update_layout(
        title=f"Predict - Candlestick Chart - Case Study #{case_number}",
        xaxis_rangeslider_visible=False
    )
    return fig

def plot_candlestick_result(df: pd.DataFrame, case_number: int):
    """
    Show full 100% of candles, mark 80% index point.
    """
    split_idx = int(len(df) * 0.8)
    vline_date = df.index[split_idx]
    last_close = df["Close"].iloc[split_idx-1]
    
    fig = go.Figure()
    # Full candles
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Result Data"
    ))
    
    # Add vertical line
    fig.add_vline(
        x=vline_date,
        line_width=1,
        line_dash="dash",
        line_color="red"
    )

    # Add horizontal ray starting from vline_date
    fig.add_shape(
        type="line",
        x0=vline_date, 
        x1=df.index[-1],       # extend to last available date
        y0=last_close, 
        y1=last_close,
        line=dict(color="blue", dash="dot", width=1),
        xref="x", yref="y"
    )
    
    fig.update_layout(
        title=f"Result - Candlestick Chart - Case Study #{case_number}",
        xaxis_rangeslider_visible=False
    )
    return fig

def compute_elo(total_preds, winrate):
    if total_preds <= 0:
        return 0
    
    # Experience factor (fast saturation for early high winrate)
    tau = 17  
    F = 1 - np.exp(-total_preds / tau) 
    
    # Dynamic winrate target
    W_target = 0.75 + 0.20 / max(1, total_preds/50)
    W_target = np.clip(W_target, 0.75, 0.95)

    # Performance factor
    G = min(1.0, winrate / W_target)

    return int(round(1000 * F * G))

def get_rank(elo):
    if elo < 200:
        return "Novice 🐣"
    elif elo < 400:
        return "Learner 📘"
    elif elo < 600:
        return "Practitioner 🔧"
    elif elo < 750:
        return "Analyst 📊"
    elif elo < 900:
        return "Pro Trader 💵"
    elif elo < 1000:
        return "Master 🧠"
    else:
        return "Grandmaster 👑"

# ------------- Streamlit Web App -------------

st.title("📊 Higher Or Lower")

# Store session variables
if "case_studies" not in st.session_state:
    st.session_state.case_studies = []
if "correct_predictions" not in st.session_state:
    st.session_state.correct_predictions = 0
if "total_predictions" not in st.session_state:
    st.session_state.total_predictions = 0
if "current_case" not in st.session_state:
    st.session_state.current_case = None
if "submitted_case_number" not in st.session_state:
    st.session_state.submitted_case_number = False
if "submitted_prediction" not in st.session_state:
    st.session_state.submitted_prediction = False
if "submitted_decision" not in st.session_state:
    st.session_state.submitted_decision = False

if st.session_state.get("current_case") is None:
    st.caption("""
    Candlestick Chart Pattern Prediction App — with up to **10,000 case studies** for practice!  
    Sharpen your intuition by predicting price direction from incomplete candlestick charts.
    """)

    st.markdown("""
    ### 🎯 How to Play
    1. **Choose a case study number** → you will receive a chart with **160 candlesticks**.  
        - A **red vertical line** marks the point where you must predict.  
        - A **blue horizontal line** shows the current closing price that will decide your result.  
    2. **Your goal** → predict if at the end of the chart(after **40 candlesticks** left) the **final closing price** will be **higher or lower** than the end of the blue line.  
    3. **Submit your prediction** → you will get the result immediately.  
    4. **Live summary** updates after each round:
        - Tracks case count  
        - Winrate  
        - An **ELO-like score** (dynamic skill rating: the more you play + the more you win, the higher your score).  

    ---

    ### ⚙️ Operations
    - ▶️ **Start Predict** : Begin your first case study.  
    - ⏭️ **Next Case** : Move on to the next study after completing one.  
    - 🛑 **See Report** : End session and see your full summary.  
    - 🔄 **Reset App** : Finalize your score, see your **ELO score + rank**, then reset the app to start fresh.
    """)

# If we started
if st.session_state.current_case is not None:

    # 2️⃣ Input Case Study Number
    case_number = st.number_input(
        "Case study number (0–10000)",
        min_value=0,
        max_value=10000,
        value=st.session_state.current_case
    )
    st.caption("ℹ️ Choose a number between 0–10000. Each case study number must be unique. Number 0 will perform random operation.")

    # Define used_numbers before decision
    used_numbers = [c["Case Number"] for c in st.session_state.case_studies]

    if st.button("Submit Case Study Number"):
        # Special rule for 0 → random assignment
        if case_number == 0:
            available_numbers = list(set(range(1, 10001)) - set(used_numbers))
            if not available_numbers:
                st.error("⚠️ No available case numbers left! All numbers between 1–10000 have been used.")
            else:
                random_case_number = random.choice(available_numbers)
                st.session_state.submitted_case_number = True
                st.session_state.case_number = random_case_number
                st.session_state.submitted_prediction = False
                st.session_state.submitted_decision = False
                st.session_state.case_data = generate_candlestick_data(st.session_state.case_number)
                st.success(f"🎲 Special Rule Activated: Random Case Study Number **{random_case_number}** selected!")

        # Normal case > 0
        else:
            if case_number in used_numbers:
                st.error("⚠️ This case study number has already been used. Please submit another number.")
            else:
                st.session_state.submitted_case_number = True
                st.session_state.case_number = case_number
                st.session_state.submitted_prediction = False
                st.session_state.submitted_decision = False
                st.session_state.case_data = generate_candlestick_data(st.session_state.case_number)
                st.success(f"✅ Case Study Number {case_number} submitted successfully!")

    # 3️⃣ Show Case Study Question only after submitting number
    if st.session_state.submitted_case_number:
        st.subheader("Case Study Question")
        st.caption("ℹ️ Now is your time to predict if the chart would go higher or lower than the blue line.")
        st.plotly_chart(plot_candlestick_predict(st.session_state.case_data, st.session_state.case_number))

        # 4️⃣ Prediction Input
        prediction = st.radio(
            "Prediction", 
            ["Higher", "Lower"], 
            key=f"prediction_{st.session_state.current_case}"
        )

        if st.button("Submit Prediction"):
            # Check if this case was already solved
            solved_cases = [c["Case Number"] for c in st.session_state.case_studies]
            if st.session_state.case_number in solved_cases:
                st.warning("⚠️ This case has already been solved. Please click **Next Case** in the sidebar to continue.")
            else:
                st.session_state.submitted_prediction = True
                st.session_state.prediction = prediction

                # ✅ Use existing case_data (generated at case submit) for results
                df_case = st.session_state.case_data
                split_idx = int(len(df_case) * 0.8)

                # Actual outcome direction: last close vs last known close
                st.session_state.result_direction = (
                    "Higher" if df_case["Close"].iloc[-1] > df_case["Close"].iloc[split_idx-1]
                    else "Lower"
                )

                # Determine correctness
                if st.session_state.prediction == st.session_state.result_direction:
                    status = "✅ Correct Prediction!"
                    st.session_state.correct_predictions += 1
                else:
                    status = "❌ Wrong Prediction!"
                st.session_state.total_predictions += 1
                st.session_state.status = status

                # ✅ Save the case (only once)
                st.session_state.case_studies.append({
                    "Case Number": st.session_state.case_number,
                    "Prediction": st.session_state.prediction,
                    "Actual": st.session_state.result_direction,
                    "Status": st.session_state.status
                })

                # 5️⃣ Plot the Result chart and print outcome
                st.subheader("Case Study Result")
                st.plotly_chart(plot_candlestick_result(
                    df_case, st.session_state.case_number
                ))

                st.info(f"📢 Actual Direction: **{st.session_state.result_direction}**")
                st.success(st.session_state.status)

# ---- Sidebar Controls ----
st.sidebar.subheader("⚙️ Controls")

# 1️⃣ Start Button
if st.sidebar.button("✅ Start Predict"):
    st.session_state.current_case = 1 if st.session_state.current_case is None else st.session_state.current_case
    st.session_state.submitted_case_number = False
    st.session_state.submitted_prediction = False
    st.session_state.submitted_decision = False

# Next Case button - only active after finishing a case
if st.session_state.get("submitted_prediction", False):  # only show if result was revealed
    if st.sidebar.button("⏭️ Next Case"):
        st.session_state.current_case += 1
        st.session_state.submitted_case_number = False
        st.session_state.submitted_prediction = False
        st.session_state.submitted_decision = False
        st.rerun()

# Reset app button in sidebar
if st.sidebar.button("🔄 Reset App"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.sidebar.success("App data cleared. Restarting...")
    st.rerun()

# ---- Sidebar Live Summary ----
st.sidebar.subheader("📊 Live Summary")

if "case_studies" in st.session_state and len(st.session_state.case_studies) > 0:
    df_summary = pd.DataFrame(st.session_state.case_studies)

    st.sidebar.write("**Number of case studies:**", len(df_summary))
    st.sidebar.write("**Correct predictions:**", st.session_state.correct_predictions)
    st.sidebar.write("**Total predictions:**", st.session_state.total_predictions)

    # Calculate winrate
    winrate = (
        st.session_state.correct_predictions / st.session_state.total_predictions
        if st.session_state.total_predictions > 0 else 0
    )
    winrate_percent = round(winrate * 100, 2)

    # Calculate ELO
    elo = compute_elo(st.session_state.total_predictions, winrate)
    rank = get_rank(elo)

    st.sidebar.markdown(f"**Winrate:** {winrate_percent}%")
    st.sidebar.markdown(f"🏅 **ELO Score:** {elo}")
    st.sidebar.markdown(f"🎖️ **Rank:** {rank}")

    # Show small case history
    st.sidebar.write("**Recent Cases:**")
    st.sidebar.table(df_summary.tail(5)[["Case Number", "Prediction", "Actual", "Status"]])
else:
    st.sidebar.info("No case studies yet. Start first!")

