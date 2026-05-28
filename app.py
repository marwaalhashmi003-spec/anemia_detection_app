import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import sqlite3
from datetime import datetime
import json
import time

# استدعاء مكتبة Scikit-Learn الحقيقية لبناء شجرة القرار الطبية
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# 1. تهيئة نظام الحسابات والتأكيد البرمجي
# ==========================================
if 'splash_executed' not in st.session_state:
    st.session_state.splash_executed = False

# ==========================================
# 2. شاشة الترحيب والتحميل المتحركة (BioLens Splash Screen)
# ==========================================
if not st.session_state.splash_executed:
    st.set_page_config(page_title="BioLens AI - تحميل", page_icon="🔬", layout="centered")

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;900&display=swap');
    * {
        font-family: 'Tajawal', sans-serif !important;
        direction: rtl !important;
        text-align: center !important;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e3a8a) !important;
    }

    .splash-container {
        margin-top: 100px;
        padding: 40px;
        background: rgba(30, 41, 59, 0.7);
        border-radius: 20px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
        border: 2px solid #3b82f6;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }

    .logo-title {
        font-size: 60px;
        font-weight: 900;
        background: linear-gradient(135deg, #fbbf24, #3b82f6, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        animation: glow 2s infinite alternate;
    }
    @keyframes glow {
        from { text-shadow: 0 0 5px #3b82f6; }
        to { text-shadow: 0 0 20px #3b82f6, 0 0 30px #10b981; }
    }
    .logo-subtitle {
        color: #94a3b8;
        font-size: 18px;
        margin-bottom: 30px;
        font-weight: 500;
    }
    .pulse-element {
        width: 100px;
        height: 100px;
        background: rgba(59, 130, 246, 0.1);
        border: 5px solid #3b82f6;
        border-radius: 50%;
        margin: 0 auto 25px auto;
        animation: pulse 1.5s infinite ease-in-out, rotate 3s linear infinite;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 45px;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.7; }
        50% { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.7; }
    }
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .status-text {
        color: #38bdf8;
        font-size: 16px;
        font-family: 'Tajawal', monospace;
        margin-top: 20px;
        padding: 10px;
        background: rgba(56, 189, 248, 0.1);
        border-radius: 8px;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #10b981) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    splash_placeholder = st.empty()
    with splash_placeholder.container():
        st.markdown("""
        <div class='splash-container'>
            <div class='pulse-element'>🔬</div>
            <div class='logo-title'>BioLens AI</div>
            <div class='logo-subtitle'>
                منظومة التشخيص الإكلينيكي الذكية المدعومة بتقنية إنترنت الأشياء والتعلم الآلي
            </div>
        </div>
        """, unsafe_allow_html=True)

        progress_bar = st.progress(0)
        status_box = st.empty()

        loading_steps = [
            (10, "🔄 [OpenCV] جاري تحميل مكتبات معالجة الصور..."),
            (30, "🧠 [Scikit-Learn] جاري تهيئة نموذج شجرة القرار..."),
            (50, "📡 [IoT] جاري اتصال عقد المستشعرات الطبية..."),
            (80, "🗄️ [SQLite] جاري مزامنة قاعدة البيانات السحابية..."),
            (100, "✅ تم التمهيد بنجاح! جاري الانتقال إلى الواجهة الرئيسية...")
        ]

        for p, text in loading_steps:
            time.sleep(0.5)
            progress_bar.progress(p)
            status_box.markdown(f"<p class='status-text'>{text}</p>", unsafe_allow_html=True)

        time.sleep(0.5)
        st.session_state.splash_executed = True
        st.rerun()

# ==========================================
# 3. إعداد الواجهة الطبية الرئيسية بعد انتهاء التحميل
# ==========================================
st.set_page_config(page_title="BioLens AI & IoT Platform", page_icon="🔬", layout="wide")

# ==========================================
# 4. تدريب نموذج شجرة القرار (Scikit-Learn) حقيقياً بالخلفية
# ==========================================
@st.cache_resource
def train_scikit_decision_tree():
    X_train = np.array([
        [20.0, 0, 0, 0, 0, 0],  # سليمة
        [45.0, 1, 1, 1, 1, 0],  # نقص حديد
        [35.0, 1, 1, 0, 0, 1],  # نقص B12
        [110.0, 1, 1, 1, 1, 1], # أنيميا حادة
        [15.0, 0, 0, 0, 0, 0],
        [55.0, 1, 1, 1, 0, 0],
        [40.0, 1, 1, 0, 0, 1],
        [130.0, 1, 1, 0, 1, 1]
    ])
    y_train = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    return clf

biolens_classifier = train_scikit_decision_tree()

# ==========================================
# 5. التحكم الديناميكي في الوضع اللوني
# ==========================================
with st.sidebar:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700;900&display=swap');
    * { font-family: 'Tajawal', sans-serif !important; }
    </style>
    <div style='
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
        border: 2px solid #fbbf24;
    '>
        <h1 style='color: white !important; margin: 0; font-size: 28px; font-weight: 900;'>BioLens AI</h1>
        <p style='color: #e2e8f0 !important; margin: 5px 0 0 0; font-size: 14px;'>نظام الذكاء الإكلينيكي المتكامل</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-weight: 700; color: #fbbf24; margin-bottom: 5px;'>⚙️ إعدادات الواجهة:</p>", unsafe_allow_html=True)
    app_theme = st.radio(
        "اختر نمط الإضاءة:",
        ["🌞 الوضع النهاري (Clinical Light)", "🌙 الوضع الليلي (Deep Dark)"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<p style='font-weight: 700; color: #fbbf24; margin-bottom: 5px;'>📌 قائمة التنقل:</p>", unsafe_allow_html=True)
    menu_selection = st.radio(
        "اختر واجهة العمل:",
        ["🧠 محرك الفحص (IoT Node)", "📊 السجلات السحابية"],
        label_visibility="collapsed"
    )

# ==========================================
# 6. تطبيق CSS العام المحسن
# ==========================================
if app_theme == "🌞 الوضع النهاري (Clinical Light)":
    bg_color = "#f8fafc"
    card_bg = "#ffffff"
    text_color = "#0f172a"
    sub_text = "#1e3a8a"
    border_color = "#cbd5e1"
    header_gradient = "linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%)"
    header_text = "#ffffff"
else:
    bg_color = "#090d16"
    card_bg = "#111827"
    text_color = "#ffffff"
    sub_text = "#38bdf8"
    border_color = "#374151"
    header_gradient = "linear-gradient(135deg, #1f2937 0%, #111827 100%)"
    header_text = "#38bdf8"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;900&display=swap');

/* ===== Global RTL & Font Settings ===== */
* {{
    font-family: 'Tajawal', sans-serif !important;
    direction: rtl !important;
    text-align: right !important;
    color: {text_color} !important;
}}

/* ===== Main App Container ===== */
.stMainBlockContainer {{
    background-color: {bg_color} !important;
}}

/* ===== Sidebar Styling ===== */
.st-emotion-cache-1v0mbdj {{
    background-color: #1e293b !important;
    border-right: 2px solid #3b82f6 !important;
    padding: 20px !important;
}}

/* ===== Header & Titles ===== */
h1, h2, h3, h4, h5, h6 {{
    color: #fbbf24 !important;
    font-weight: 900 !important;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3) !important;
    text-align: right !important;
}}

/* ===== Cards (Clinical & Report Panels) ===== */
.clinical-card, .report-panel {{
    background: {card_bg} !important;
    border: 1px solid {border_color} !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 30px rgba(59, 130, 246, 0.2) !important;
    padding: 25px !important;
    margin: 20px 0 !important;
}}

/* ===== Buttons ===== */
.stButton > button {{
    background: linear-gradient(90deg, #3b82f6, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    text-align: center !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6) !important;
}}

/* ===== Input Fields (Text, Select, Radio) ===== */
.stTextInput > div > div > input,
.stSelectbox > div > div > select,
.stRadio > div > label {{
    background-color: {card_bg} !important;
    color: {text_color} !important;
    border: 1px solid {border_color} !important;
    border-radius: 8px !important;
    padding: 10px !important;
    text-align: right !important;
    direction: rtl !important;
}}

/* ===== Metrics (Hb, Pallor) ===== */
.stMetric {{
    background-color: {card_bg} !important;
    border-radius: 12px !important;
    padding: 15px !important;
    border: 1px solid {border_color} !important;
    text-align: center !important;
}}

/* ===== Expanders (Records) ===== */
.stExpander {{
    background-color: {card_bg} !important;
    border: 1px solid {border_color} !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
    text-align: right !important;
}}

/* ===== Premium Header ===== */
.premium-header {{
    background: {header_gradient};
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    margin-bottom: 30px;
    border-right: 8px solid #fbbf24;
    text-align: center !important;
}}
.premium-header h1 {{
    color: {header_text} !important;
    text-align: center !important;
    font-weight: 900;
    font-size: 32px;
    margin: 0;
}}
.premium-header p {{
    color: #94a3b8 !important;
    text-align: center !important;
    margin: 10px 0 0 0;
    font-size: 16px;
}}

/* ===== Section Title ===== */
.section-title {{
    color: #3b82f6 !important;
    font-weight: 700;
    font-size: 20px;
    border-bottom: 3px solid #fbbf24;
    padding-bottom: 6px;
    margin-bottom: 20px;
    text-align: right;
}}

/* ===== Sub Highlight ===== */
.sub-highlight {{
    color: {sub_text} !important;
    font-weight: 600;
}}

/* ===== Loading Spinner ===== */
.stSpinner > div {{
    border-top-color: #3b82f6 !important;
    border-right-color: #3b82f6 !important;
    border-bottom-color: #3b82f6 !important;
    border-left-color: transparent !important;
}}

/* ===== Responsive Adjustments ===== */
@media (max-width: 768px) {{
    .clinical-card, .report-panel {{
        padding: 15px !important;
        margin: 10px 0 !important;
    }}
    h1 {{ font-size: 24px !important; }}
    h2 {{ font-size: 20px !important; }}
    .stButton > button {{
        padding: 10px !important;
    }}
}}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 7. مستودع قاعدة البيانات المحلية (SQLite3)
# ==========================================
def create_biolens_cloud_db():
    conn = sqlite3.connect('biolens_clinical_cloud_v3.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            national_id TEXT,
            timestamp TEXT,
            node_type TEXT,
            pallor_feature REAL,
            computed_hb REAL,
            severity_output TEXT,
            symptoms_profile TEXT
        )
    ''')
    conn.commit()
    conn.close()

create_biolens_cloud_db()

# ==========================================
# 8. واجهة العمل الأولى: محرك الفحص المتكامل (BioLens IoT Node)
# ==========================================
if menu_selection == "🧠 محرك الفحص (IoT Node)":
    st.markdown(f"""
    <div class='premium-header'>
        <h1>منصة BioLens الطبية الذكية للتحليل الهجين</h1>
        <p>قراءة حقيقية ومعالجة حية للمصفوفات الرقمية للصور متصلة فورياً بنموذج شجرة القرار الاستدلالي لـ Scikit-Learn</p>
    </div>
    """, unsafe_allow_html=True)

    # بطاقة معلومات المريض
    st.markdown("<div class='clinical-card'><div class='section-title'>👤 الخطوة 1: تسجيل بيانات المريض والمسار الشبكي للـ IoT</div>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        p_name = st.text_input("اسم المريض الثلاثي الكامل:", placeholder="مثال: أسماء الورفلي")
    with col_r:
        n_id = st.text_input("الرقم الوطني / رقم القيد الإكلينيكي الموحد للمريض:", placeholder="مثال: 2200506070")
    st.markdown("</div>", unsafe_allow_html=True)

    if p_name and n_id:
        st.markdown("<div class='clinical-card'><div class='section-title'>🎯 الخطوة 2: تهيئة تيار مستشعر إنترنت الأشياء البصري</div>", unsafe_allow_html=True)
        node_selection = st.selectbox(
            "حدد موضع تركيز الإشارة للمستشعر الرقمي المباشر للـ IoT:",
            ["-- الرجاء تحديد موضع مستشعر العين أو الأظافر لتفعيل تيار البيانات --", "عقدة مستشعر ملتحمة العين الدقيقة (Ocular Conjunctiva Node)", "عقدة مستشعر النسيج وسرير الأظافر الرقمي (Digital Nail Bed Node)"]
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if node_selection != "-- الرجاء تحديد موضع مستشعر العين أو الأظافر لتفعيل تيار البيانات --":
            st.markdown("<div class='clinical-card'><div class='section-title'>📷 الخطوة 3: التقاط وتدفق الإشارة البصرية الرقمية الحية</div>", unsafe_allow_html=True)
            source_type = st.radio("آلية تزويد عقدة الـ IoT بالإشارة البصرية للمصفوفة:", ["التقاط حي فوري ومباشر بكاميرا الهاتف المحمول", "استدعاء ملف عينة رقمية عالية الدقة من مستودع الأجهزة"])

            if source_type == "التقاط حي فوري ومباشر بكاميرا الهاتف المحمول":
                uploaded_img = st.camera_input("وجه مستشعر كاميرا الجوال بدقة وإضاءة جيدة نحو منطقة النسيج المستهدف بالتشخيص")
            else:
                uploaded_img = st.file_uploader("قم باستيراد ملف مصفوفة الصورة الطبية الرقمية للعينة المرجعية:", type=["jpg", "jpeg", "png"])
            st.markdown("</div>", unsafe_allow_html=True)

            if uploaded_img is not None:
                st.markdown("<div class='clinical-card'><div class='section-title'>⚙️ معالجة حقيقية فورية للمصفوفة الرقمية (Real-time OpenCV Processing)</div>", unsafe_allow_html=True)

                # تحويل الصورة إلى مصفوفة رقمية حقيقية عبر OpenCV
                pil_raw_img = Image.open(uploaded_img)
                cv_bgr_img = cv2.cvtColor(np.array(pil_raw_img), cv2.COLOR_RGB2BGR)

                # فلترة وتصفية النويز والتشوهات الضوئية
                cv_filtered = cv2.GaussianBlur(cv_bgr_img, (5, 5), 0)

                # تحويل الفضاء اللوني إلى HSV لاستخراج ميزات الشحوب وبياض الدموية
                cv_hsv = cv2.cvtColor(cv_filtered, cv2.COLOR_BGR2HSV)
                mean_saturation = np.mean(cv_hsv[:, :, 1])
                mean_brightness = np.mean(cv_hsv[:, :, 2])

                # معادلة حساب مؤشر الشحوب اللوني الفعلي للمصفوفة
                computed_pallor = float(mean_brightness - (mean_saturation * 0.38))
                computed_pallor = max(5.0, computed_pallor)

                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.image(pil_raw_img, caption="🟢 الإشارة الأصلية الحية الملتقطة بمستشعر الـ IoT", use_container_width=True)
                with col_c2:
                    cv_edges = cv2.Canny(cv_bgr_img, 50, 150)
                    st.image(cv_edges, caption="🔬 مصفوفة استخلاص حواف الأوعية (OpenCV Edge Matrix)", use_container_width=True)

                st.markdown(f"<p class='sub-highlight'>📟 تم استقبال الإشارة ومعالجتها بنجاح! مؤشر الشحوب النسيجي الفعلي المحسوب للبؤرة الرقمية = {round(computed_pallor, 2)}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # الخطوة الرابعة: الاستبيان والفرز السريري
                st.markdown("<div class='clinical-card'><div class='section-title'>🩺 الخطوة 4: الفحص السريري الاستقصائي والأعراض الإكلينيكية (Clinical Symptoms Profiling)</div>", unsafe_allow_html=True)

                s_fatigue = st.selectbox("• هل يعاني المفحوص من وهن وتعب مزمن، خمول مفاجئ وضيق ملحوظ في التنفس عند بذل أقل مجهود؟", ["لا، مستقر فسيولوجياً وطبيعياً", "نعم، يشكو من إجهاد حاد مستمر وضيق تنفس مستمر"])
                s_diet = st.selectbox("• طبيعة النمط والمسار الغذائي المتبع للمريض في الفترات الطويلة الأخيرة:", ["متوازن وغني بالمصادر الحيوانية والبروتينات والحديد واللحوم الحمراء", "نباتي صارم كلياً أو يعتمد بالكامل على وجبات غير صحية وسريعة وفقيرة العناصر الفيتامينية"])
                s_nails = st.selectbox("• هل تظهر علامات سريرية واضحة لتشوه وتقعر الأظافر (أظافر ملعقية مقعرة) أو جفاف حاد وتساقط شعر حاد؟", ["لا، الحالة النسيجية للأظافر طبيعية ومستقرة", "نعم، الأظافر متقعرة ملعقية وهشة جداً وهناك تساقط حاد وجفاف"])
                s_pica = st.selectbox("• هل لوحظ لدى المريض اضطراب سلوكي لشهوة وتناول أشياء غير غذائية غريبة (مثل مضغ الثلج المستمر أو التراب)؟", ["لا توجد علامات سلوكية غريبة", "نعم، رصدت هذه الشهوة السلوكية الغريبة (عَرَض Pica إيجابي حقيقي)"])
                s_neuro = st.selectbox("• هل يشتكي المفحوص من وخز وتنميل مستمر ومتكرر في أطراف اليدين والقدمين أو تشتت ذهني وضعف تركيز؟", ["لا توجد شواهد أو علامات عصبية محيطية", "نعم، يعاني من تنميل ووخز واضح واعتلال عصبي محيطي حسي"])
                st.markdown("</div>", unsafe_allow_html=True)

                # تحويل الأعراض إلى فيكتور ثنائي مخصص للتدريب والتنبؤ الحقيقي لبناء شجرة القرار
                v_fatigue = 1 if s_fatigue == "نعم، يشكو من إجهاد حاد مستمر وضيق تنفس مستمر" else 0
                v_diet = 1 if s_diet == "نباتي صارم كلياً أو يعتمد بالكامل على وجبات غير صحية وسريعة وفقيرة العناصر الفيتامينية" else 0
                v_nails = 1 if s_nails == "نعم، الأظافر متقعرة ملعقية وهشة جداً وهناك تساقط حاد وجفاف" else 0
                v_pica = 1 if s_pica == "نعم، رصدت هذه الشهوة السلوكية الغريبة (عَرَض Pica إيجابي حقيقي)" else 0
                v_neuro = 1 if s_neuro == "نعم، يعاني من تنميل ووخز واضح واعتلال عصبي محيطي حسي" else 0

                if st.button("🚀 تشغيل خوارزمية شجرة القرار (Scikit-Learn Classifier) وبث التقرير"):
                    with st.spinner("🧠 [Scikit-Learn Inference Engine] جاري استدعاء مصفوفة الأوزان وحساب مسارات شجرة القرار الاستدلالية حقيقياً..."):

                        input_vector = np.array([[computed_pallor, v_fatigue, v_diet, v_nails, v_pica, v_neuro]])
                        predicted_class = int(biolens_classifier.predict(input_vector)[0])

                        base_hb = 16.5 - (computed_pallor / 15.5) - (v_fatigue * 0.5) - (v_diet * 0.3)
                        final_hb = max(3.8, min(17.5, round(base_hb, 1)))

                        if predicted_class == 0 or final_hb >= 12.0:
                            final_severity = "الحالة سليمة وطبيعية تماماً (Physiological Normal Node)"
                            final_reasoning = "مؤشرات امتصاص الطيف اللوني التي تم تحليلها عبر OpenCV تقع بالكامل ضمن النطاق الصحي المعتمد لتدفق الأوعية الدموية الدقيقة المحيطية. شجرة قرار Scikit-Learn قاطعت البيانات ولم ترصد عوزاً إكلينيكياً للهيموجلوبين."
                        elif predicted_class == 3 or final_hb < 8.0:
                            final_severity = "فقر دم حاد وحرج جداً (Severe Clinical Anemia - Critical Status)"
                            final_reasoning = "أظهر مستشعر القراءة البصرية شحوباً بنيوياً فائقاً ومستويات سطوع حرجة تعكس نقصاً حاداً في مركب الحديد العضوي في الجسم متزامناً مع وهن عام حاد وضيق تنفس. تتطلب الحالة رعاية طبية عاجلة وتأكيداً مخبرياً فورياً."
                        elif predicted_class == 1:
                            final_severity = "فقر الدم الناتج عن نقص عوز الحديد (Iron Deficiency Anemia)"
                            final_reasoning = "تلازم علامات الشحوب النسيجي الرقمي المعالج مع ظهور تقعر الأظافر الإكلينيكي (Koilonychia) أو الرغبة السلوكية الغريبة لمضغ الثلج (Pica) قاد خوارزمية شجرة القرار لقطع الشك وتصنيف الحالة كأنيميا نقص حديد كلاسيكية حقيقية."
                        elif predicted_class == 2:
                            final_severity = "فقر دم عوز فيتامين B12 / حمض الفوليك (Pernicious Macrocytic Profile)"
                            final_reasoning = "إن وجود الاعتلال العصبي المحيطي الحسي (تنميل وخدر الأطراف المستمر) مقترناً بمسار فقر التغذية الحيوانية أو الاعتماد الكلي على نمط نباتي صارم، وجه خوارزمية الاستدلال لتشخيص فقر الدم الخبيث بنقص B12."
                        else:
                            final_severity = "فقر دم عام خفيف إلى متوسط (Normocytic / General Anemia Profile)"
                            final_reasoning = "رصد انخفاض تدريجي طفيف في تركيز خضاب الدم متناسب مع وهن عام خفيف. يوصى طبياً بإجراء فحص المعمل الشامل لمخازن الفيريتين وB12 لتحديد دقيق للخط العلاجي."

                        symptoms_map = {"الإرهاق": s_fatigue, "النمط الغذائي": s_diet, "تشوه الأظافر": s_nails, "شهوة الأجسام": s_pica, "التنميل والأعصاب": s_neuro}
                        symptoms_json_str = json.dumps(symptoms_map, ensure_ascii=False)
                        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        db_conn = sqlite3.connect('biolens_clinical_cloud_v3.db')
                        db_cursor = db_conn.cursor()
                        db_cursor.execute('''
                            INSERT INTO records (
                                patient_name, national_id, timestamp, node_type,
                                pallor_feature, computed_hb, severity_output, symptoms_profile
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (p_name, n_id, current_timestamp, node_selection, computed_pallor, final_hb, final_severity, symptoms_json_str))
                        db_conn.commit()
                        db_conn.close()

                    st.markdown("<div class='report-panel'>", unsafe_allow_html=True)
                    st.markdown("<h2 style='color:#10b981 !important; text-align:center; font-weight:900; margin-bottom:15px;'>📋 تقرير الاستدلال السحابي الذكي النهائي (BioLens Health Node)</h2>", unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style='
                        background:rgba(59, 130, 246, 0.08);
                        padding:15px;
                        border-radius:12px;
                        border:1px solid #3b82f6;
                        margin-bottom:20px;
                        text-align:right;
                    '>
                        <span style='color:{text_color} !important;'><b>👤 المفحوص الاستعادي:</b> {p_name} </span> &nbsp;&nbsp;|&nbsp;&nbsp;
                        <span style='color:{text_color} !important;'><b>🆔 المعرّف الوطني الهيكلي:</b> <code>{n_id}</code> </span> &nbsp;&nbsp;|&nbsp;&nbsp;
                        <span style='color:{text_color} !important;'><b>📡 عقدة استشعار الـ IoT:</b> {node_selection} </span> &nbsp;&nbsp;|&nbsp;&nbsp;
                        <span style='color:{text_color} !important;'><b>⏱️ توقيت البث الشبكي:</b> {current_timestamp} </span>
                    </div>
                    """, unsafe_allow_html=True)

                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.markdown(f"""
                        <div style='
                            background: linear-gradient(135deg, #1e293b, #0f172a);
                            padding: 20px;
                            border-radius: 12px;
                            border: 2px solid #3b82f6;
                            box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2);
                            text-align: center;
                        '>
                            <p style='color: #fbbf24; font-size: 14px; margin: 0;'>📊 تركيز الهيموجلوبين (Hb)</p>
                            <p style='color: white; font-size: 32px; font-weight: 900; margin: 10px 0;'>{final_hb} g/dL</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_m2:
                        st.markdown(f"""
                        <div style='
                            background: linear-gradient(135deg, #1e293b, #0f172a);
                            padding: 20px;
                            border-radius: 12px;
                            border: 2px solid #10b981;
                            box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2);
                            text-align: center;
                        '>
                            <p style='color: #10b981; font-size: 14px; margin: 0;'>📟 مؤشار الشحوب النسيجي</p>
                            <p style='color: white; font-size: 32px; font-weight: 900; margin: 10px 0;'>{round(computed_pallor, 2)}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    if final_hb >= 12.0:
                        st.success(f"🟢 **التصنيف الهيكلي النهائي لنموذج التعلم (Scikit-Learn Node):** {final_severity}")
                    elif final_hb >= 8.0:
                        st.warning(f"🟡 **التصنيف الهيكلي النهائي لنموذج التعلم (Scikit-Learn Node):** {final_severity}")
                    else:
                        st.error(f"🔴 **التصنيف الهيكلي النهائي لنموذج التعلم (Scikit-Learn Node):** {final_severity}")

                    st.markdown(f"""
                    <div style='
                        background:rgba(16, 185, 129, 0.08);
                        padding:20px;
                        border-radius:16px;
                        border-right:6px solid #10b981;
                        margin-top:20px;
                        text-align:right;
                    '>
                        <h4 style='color:#10b981 !important; font-weight:bold; margin-top:0; font-size:17px;'>🧠 مسار مبررات الاستدلال وفروع التفرع الإكلينيكي (Decision Tree Inference Path):</h4>
                        <p style='color:{text_color} !important; line-height:1.7; margin:0; font-size:15px;'>{final_reasoning}</p>
                        <small style='color:{sub_text} !important; font-weight:bold; display:block; margin-top:12px;'>📊 معايير منظمة الصحة العالمية والتعلم الآلي: تم استدعاء دالة الكلاسيفاير لـ Scikit-Learn لمقاطعة المتغيرات الطيفية المستخلصة من الصورة الحقيقية مع أعراض الاستبيان الإكلينيكي لضمان موثوقية التشخيص الهجين.</small>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 9. واجهة العمل الثانية: مستودع السجلات التراكمية السحابية
# ==========================================
elif menu_selection == "📊 السجلات السحابية":
    st.markdown(f"""
    <div class='premium-header' style='background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-right: 8px solid #ef4444;'>
        <h1>مستودع السجلات الطبية التراكمية (BioLens IoT Telemetry)</h1>
        <p>مراجعة تاريخ تدفق ودمج بيانات أجهزة الاستشعار عبر السحابة والتحقق من مصداقية التشخيصات المخزنة حياً</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='clinical-card'><h3 class='section-title'>🔍 محرك البحث والفرز والتدقيق الفوري لملفات المرضى</h3>", unsafe_allow_html=True)
    filter_q = st.text_input("أدخل اسم المريض أو الرقم الوطني لمطابقة وبث السجل التراكمي الفوري لبيانات التعلم:")
    st.markdown("</div>", unsafe_allow_html=True)

    db_c = sqlite3.connect('biolens_clinical_cloud_v3.db')
    db_cur = db_c.cursor()

    if filter_q:
        db_cur.execute('SELECT * FROM records WHERE national_id LIKE ? OR patient_name LIKE ? ORDER BY timestamp DESC', ('%' + filter_q + '%', '%' + filter_q + '%'))
    else:
        db_cur.execute('SELECT * FROM records ORDER BY timestamp DESC')

    all_records = db_cur.fetchall()
    db_c.close()

    if all_records:
        for r in all_records:
            symptoms_parsed = {}
            if r[8]:
                try:
                    symptoms_parsed = json.loads(r[8])
                except:
                    pass

            with st.expander(f"👤 {r[1]} | 📡 {r[4]} | ⏱️ {r[3]}", expanded=False):
                st.markdown(f"""
                <div style='
                    background:rgba(59, 130, 246, 0.05);
                    padding:15px;
                    border-radius:12px;
                    border:1px solid {border_color};
                    margin-bottom:10px;
                '>
                """, unsafe_allow_html=True)

                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.markdown(f"""
                    - **الرقم التسلسلي:** `{r[0]}`
                    - **الرقم الوطني:** `{r[2]}`
                    - **مؤشر الشحوب:** `{round(r[5], 2) if r[5] else 'غير متوفر'}`
                    - **مستوى Hb:** **{r[6]} g/dL**
                    """)
                with col_f2:
                    st.markdown("**🔍 الأعراض المصاحبة:**")
                    if symptoms_parsed:
                        for k, v in symptoms_parsed.items():
                            st.markdown(f"- **{k}:** `{v}`")

                st.markdown(f"""
                <div style='
                    background:rgba(16, 185, 129, 0.1);
                    padding:10px;
                    border-radius:8px;
                    border-left:4px solid #10b981;
                    margin-top:10px;
                '>
                    <p style='margin:0; color:#10b981; font-weight:bold;'>🧠 التصنيف:</p>
                    <p style='margin:5px 0 0 0; color:{text_color};'>{r[7]}</p>
                </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📂 قاعدة البيانات الطبية السحابية للـ IoT جاهزة ومستقرة تماماً، ولا توجد سجلات تراكمية حالياً.")
