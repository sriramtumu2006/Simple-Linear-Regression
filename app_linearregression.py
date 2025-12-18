
# Page Config #
st.set_page_config("Linear Regression",layout="centered")

# Load CSS # 
def load_css(filename):
    with open(filename) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
load_css("style.css")

# Title #
st.markdown("""
            <div class="card">
            <h1> Linear Regression</h1>
            <p>Predict <b>Tip Amount</b> from <b>Total Bill</b> using Linear Regression...</p>
            </div>
            """,unsafe_allow_html=True)

# Load Data #
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df=load_data()

# Dataset Preview #
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>',unsafe_allow_html=True)

# Prepare Data #
x,y=df[["total_bill"]],df["tip"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# Train Model #
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

# Metrics #
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
adjusted_r2=1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - 2)

# Visualization #
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Total Bill vs Tip")
fig,ax=plt.subplots()
ax.scatter(df["total_bill"],df["tip"],color="#744ee7",alpha=0.7)
ax.plot(df["total_bill"],model.predict(scaler.transform(df[["total_bill"]])),color="#f74141")
ax.set_xlabel("Total Bill ($)")
ax.set_ylabel("Tip ($)")
st.pyplot(fig)
st.markdown('</div>',unsafe_allow_html=True)

# Performance Metrics #
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Model Performance")
c1,c2,c3,c4=st.columns(4)
c1.metric("Mean Absolute Error (MAE)",f"{mae:.2f}")
c2.metric("Root Mean Squared Error (RMSE)",f"{rmse:.2f}")
c3.metric("R² Score",f"{r2:.2f}")
c4.metric("Adjusted R² Score",f"{adjusted_r2:.2f}")
st.markdown('</div>',unsafe_allow_html=True)

# m & c #
st.markdown("""
            <div class="card">
            <h3>Model Coefficients</h3>
            <p>Slope (m): <b>{:.4f}</b></p>
            <p>Intercept (c): <b>{:.4f}</b></p>
            </div>
            """.format(model.coef_[0],model.intercept_),unsafe_allow_html=True)

# Prediction #
st.markdown('<div class="card">',unsafe_allow_html=True)
bill_amount=st.slider("Total Bill Amount ($)",min_value=float(df.total_bill.min()),max_value=float(df.total_bill.max()),value=30.0)
tip=model.predict(scaler.transform([[bill_amount]]))[0]
st.markdown(f'<p class="prediction-box">Predicted Tip Amount: <b>${tip:.2f}</b></p>',unsafe_allow_html=True)

st.markdown('</div>',unsafe_allow_html=True)
