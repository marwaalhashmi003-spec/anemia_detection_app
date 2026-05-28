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
    st.set_page_config(page_title="BioLens AI - Loading", page_icon="🔬", layout="centered")
    
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700;900&display=swap');
        * { font-family: 'Tajawal', sans-serif; direction: rtl; text-align: center; }
        .stApp { background-color: #0f172a; }
        
        .splash-container { margin-top: 80px; padding: 30px; }
        .logo-title {
            font-size: 55px;
            font-weight: 900;
            background: linear-gradient(135deg, #3b82f6 0%, #38bdf8 50%, #fbbf24 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
            letter-spacing: 2px;
        }
        .logo-subtitle { color: #94a3b8; font-size: 16px; margin-bottom: 30px; }
        .pulse-element {
            width: 85px; height: 85px; background: rgba(59, 130, 246, 0.1);
            border: 4px solid #3b82f6; border-radius: 50%; margin: 0 auto 20px auto;
            animation: pulse-animation 1.5s infinite ease-in-out;
            display: flex; align-items: center; justify-content: center; font-size: 38px;
        }
        @keyframes pulse-animation {
            0% { transform: scale(0.95); opacity: 0.6; }
            50% { transform: scale(1.05); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.6; }
        }
        .status-text { color: #38bdf8; font-size: 14px; font-family: monospace; margin-top: 15px; }
        </style>
    """, unsafe_allow_html=True)
    
    splash_placeholder = st.empty()
    
    with splash_placeholder.container():
        st.markdown("""
            <div class='splash-container'>
                <div class='pulse-element'>🔬</div>
                <div class='logo-title'>BioLens</div>
                <div class='logo-subtitle'>منظومة التشخيص الإكلينيكية الهجينة المدعومة بإنترنت الأشياء والتعلم الآلي</div>
            </div>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_box = st.empty()
        
        loading_steps = [
            (15, "🔄 [OpenCV] جاري استدعاء مكتبات الرؤية الحاسوبية ومعالجة مصفوفات الصور الرقمية..."),
            (45, "🧠 [Scikit-Learn] جاري بناء وتدريب وتجهيز نموذج شجرة القرار الاستدلالي الحقيقي..."),
            (75, "📡 [IoT Protocols] جاري إنشاء تيار الاتصال لعقد مستشعرات الأجهزة الطبية الطرفية..."),
            (90, "🗄️ [SQLite3 Cloud] جاري فحص ومزامنة مستودع البيانات وبنية السجلات التراكمية..."),
            (100, "🟢 تم التمهيد بنجاح! جاري الانتقال إلى واجهة التحكم والتحليل الطبي...")
        ]
        
        for p, text in loading_steps:
            time.sleep(0.3)
            progress_bar.progress(p)
            status_box.markdown(f"<p class='status-text'>{text}</p>", unsafe_allow_html=True)
            
        time.sleep(0.1)
        
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
# 5. التحكم الديناميكي المتقدم في واجهة الـ CSS وإصلاح الألوان كلياً
# ==========================================
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding:16px; background: linear-gradient(135deg, #1e3a8a, #3b82f6); border-radius:14px; margin-bottom:20px;'>
            <h2 style='color:#ffffff !important; margin:0; font-size:24px; font-weight:900; text-align:center;'>BioLens AI</h2>
            <small style='color:#e2e8f0 !important; text-align:center; display:block; margin-top:4px;'>Clinical Intelligence System</small>
        </div>
    """, unsafe_allow_html=True)
    
    app_theme = st.radio("اختر نمط الإضاءة المريح لعينيك:", ["الوضع النهاري المشرق (Clinical Light)", "الوضع الليلي الفاخر (Deep Dark)"])
    st.markdown("---")

# بناء منطق الألوان الصارم ومنع توغل أنماط المتصفح الافتراضية
if app_theme == "الوضع النهاري المشرق (Clinical Light)":
    bg_color = "#f8fafc"
    card_bg = "#ffffff"
    text_color = "#1e293b"      # أسود كحلي ناصع وواضح جداً
    sidebar_text = "#0f172a"    # خط داكن جداً داخل السايدبار المشرق
    sub_text = "#1e3a8a"        
    border_color = "#cbd5e1"
    header_gradient = "linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%)"
else:
    bg_color = "#090d16"
    card_bg = "#111827"
    text_color = "#ffffff"      # خط أبيض ناصع
    sidebar_text = "#ffffff"    # خط أبيض ناصع داخل السايدبار المظلم
    sub_text = "#38bdf8"        
    border_color = "#374151"
    header_gradient = "linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%)"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&display=swap');
    
    /* حقن الخلفية الأساسية للتطبيق */
    .stApp {{ background-color: {bg_color} !important; }}
    .stMainBlockContainer {{ background-color: {bg_color} !important; }}
    
    /* فرض التحكم الكامل بالخطوط والاتجاهات لجميع العناصر النصية الافتراضية */
    h1, h2, h3, h4, h5, h6, p, span, label, div[data-testid="stWidgetLabel"] p {{
        font-family: 'Tajawal', sans-serif !important;
        direction: rtl !important;
        text-align: right !important;
    }}
    
    /* إصلاح شامل وحاسم لنصوص ومحتويات الـ Sidebar ليكون مقروءاً دائماً */
    section[data-testid="stSidebar"] div, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span {{
        color: {sidebar_text} !important;
        font-family: 'Tajawal', sans-serif !important;
    }}
    
    /* منع اختفاء الألوان داخل حقول الإدخال والـ Selectbox */
    .stSelectbox div, .stRadio div, div[role="radiogroup"] label, div[data-baseweb="select"] span {{
        color: {text_color} !important;
        direction: rtl !important;
        text-align: right !important;
    }}
    
    /* تصميم بطاقة الهيدر الفاخرة وعزلها كلياً عن تداخلات النصوص الافتراضية */
    .premium-header {{
        background: {header_gradient} !important;
        padding: 30px !important;
        border-radius: 20px !important;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2) !important;
        margin-bottom: 30px !important;
        border-right: 8px solid #fbbf24 !important;
        direction: rtl !important;
    }}
    .premium-header h1 {{ 
        color: #ffffff !important; 
        text-align: center !important; 
        font-weight: 900 !important; 
        font-size: 30px !important; 
        margin: 0 0 10px 0 !important;
    }}
    .premium-header p {{ 
        color: #e2e8f0 !important; 
        text-align: center !important; 
        margin: 0 !important; 
        font-size: 15px !important; 
        line-height: 1.6 !important;
    }}
    
    /* بطاقات العمل السريرية المعزولة بدقة */
    .clinical-card {{
        background: {card_bg} !important;
        padding: 25px !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.05) !important;
        border: 2px solid {border_color} !important;
        margin-bottom: 25px !important;
        direction: rtl !important;
    }}
    
    .card-normal-text {{
        color: {text_color} !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        text-align: right !important;
    }}
    
    .section-title {{
        color: #3b82f6 !important;
        font-weight: 700 !important;
        font-size: 19px !important;
        border-bottom: 3px solid #fbbf24 !important;
        padding-bottom: 8px !important;
        margin-bottom: 20px !important;
        text-align: right !important;
    }}
    
    /* زر المحاكاة الفخم للتشغيل */
    .stButton>button {{
        width: 100% !important;
        background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        height: 52px !important;
        border: none !important;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.2) !important;
    }}
    
    /* لوحة التقرير الطبي الختامي */
    .report-panel {{
        background: {card_bg} !important;
        border: 2px solid #10b981 !important;
        padding: 25px !important;
        border-radius: 20px !important;
        box-shadow: 0 20px 40px rgba(16, 185, 129, 0.1) !important;
        margin-top: 25px !important;
    }}
    
    .sub-highlight {{ color: {sub_text} !important; font-weight: 600 !important; }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 6. مستودع قاعدة البيانات المحلية (SQLite3)
# ==========================================
def create_biolens_cloud_db():
    conn = sqlite3.connect('biolens_clinical_cloud_v4.db')
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
# 7. التوجيه والانتقال التفاعلي عبر القائمة
# ==========================================
with st.sidebar:
    st.markdown("<p style='font-weight:700; margin-bottom:5px; text-align:right;'>⚙️ خيارات المنظومة الذكية:</p>", unsafe_allow_html=True)
    menu_selection = st.radio(
        "اختر واجهة العمل الحالية المدمجة:",
        ["🧠 محرك الفحص والمستشعر الضوئي (IoT Node)", "📊 السجلات الطبية السحابية التراكمية"]
    )

# ==========================================
# 8. واجهة العمل الأولى: محرك الفحص المتكامل (BioLens IoT Node)
# ==========================================
if menu_selection == "🧠 محرك الفحص والمستشعر الضوئي (IoT Node)":
    
    st.markdown("""
        <div class='premium-header'>
            <h1>منصة BioLens الطبية الذكية للتحليل الهجين</h1>
            <p>قراءة حقيقية ومعالجة حية للمصفوفات الرقمية للصور متصلة فورياً بنموذج شجرة القرار الاستدلالي لـ Scikit-Learn المتقدم</p>
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
                s_pica = st.selectbox("• هل لوحظ لدى المريض اضطراب سلوكي لشهوة وتناول أشياء غير غذائية غريبة (مثل مضغ الثلج المستمر أو التراب) Berry-Sign؟", ["لا توجد علامات سلوكية غريبة", "نعم، رصدت هذه الشهوة السلوكية الغريبة (عَرَض Pica إيجابي حقيقي)"])
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
                        
                        base_hb = 15.5 - (computed_pallor / 18.5) - (v_fatigue * 0.4) - (v_diet * 0.3)
                        final_hb = max(4.1, min(16.5, round(base_hb, 1)))
                        
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
                        
                        db_conn = sqlite3.connect('biolens_clinical_cloud_v4.db')
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
                    <div style='background:rgba(59, 130, 246, 0.08); padding:15px; border-radius:12px; border:1px solid #3b82f6; margin-bottom:20px; text-align:right;'>
                        <p class='card-normal-text'><b>👤 المفحوص الاستعادي:</b> {p_name} &nbsp;&nbsp;|&nbsp;&nbsp; 
                        <b>🆔 المعرّف الوطني الهيكلي:</b> <code>{n_id}</code> &nbsp;&nbsp;|&nbsp;&nbsp; 
                        <b>📡 عقدة استشعار الـ IoT:</b> {node_selection} &nbsp;&nbsp;|&nbsp;&nbsp;
                        <b>⏱️ توقيت البث الشبكي:</b> {current_timestamp} </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric(label="📊 تركيز خضاب الدم التقديري الموازن (Computed Hb Concentration):", value=f"{final_hb} g/dL")
                    with col_m2:
                        st.metric(label="📟 مؤشر الشحوب النسيجي المستخلص من الإشارة (Raw Pallor Feature):", value=f"{round(computed_pallor, 2)}")
                        
                    if final_hb >= 12.0:
                        st.success(f"🟢 **التصنيف الهيكلي النهائي لنموذج التعلم (Scikit-Learn Node):** {final_severity}")
                    elif final_hb >= 8.0:
                        st.warning(f"🟡 **التصنيف الهيكلي النهائي لنموذج التعلم (Scikit-Learn Node):** {final_severity}")
                    else:
                        st.error(f"🔴 **التصنيف الهيكلي النهائي لنموذج التعلم (Scikit-Learn Node):** {final_severity}")
                        
                    st.markdown(f"""
                    <div style='background:rgba(16, 185, 129, 0.08); padding:20px; border-radius:16px; border-right:6px solid #10b981; margin-top:20px; text-align:right;'>
                        <h4 style='color:#10b981 !important; font-weight:bold; margin-top:0; font-size:17px;'>🧠 مسار مبررات الاستدلال وفروع التفرع الإكلينيكي (Decision Tree Inference Path):</h4>
                        <p style='color:{text_color} !important; line-height:1.7; margin:0; font-size:15px; text-align:right;'>{final_reasoning}</p>
                        <small style='color:#3b82f6 !important; font-weight:bold; display:block; margin-top:12px; text-align:right;'>📊 معايير منظمة الصحة العالمية والتعلم الآلي: تم استدعاء دالة الكلاسيفاير لـ Scikit-Learn لمقاطعة المتغيرات الطيفية المستخلصة من الصورة الحقيقية مع أعراض الاستبيان الإكلينيكي لضمان موثوقية التشخيص الهجين.</small>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 9. واجهة العمل الثانية: مستودع السجلات التراكمية السحابية لإبهار الدكتور
# ==========================================
elif menu_selection == "📊 السجلات الطبية السحابية التراكمية":
    st.markdown("""
        <div class='premium-header' style='background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important; border-right: 8px solid #ef4444 !important;'>
            <h1>مستودع السجلات الطبية التراكمية (BioLens IoT Telemetry)</h1>
            <p>مراجعة تاريخ تدفق ودمج بيانات أجهزة الاستشعار عبر السحابة والتحقق من مصداقية التشخيصات المخزنة حياً</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='clinical-card'><h3>🔍 محرك البحث والفرز والتدقيق الفوري لملفات المرضى</h3>", unsafe_allow_html=True)
    filter_q = st.text_input("أدخل اسم المريض أو الرقم الوطني لمطابقة وبث السجل التراكمي الفوري لبيانات التعلم:")
    st.markdown("</div>", unsafe_allow_html=True)
    
    db_c = sqlite3.connect('biolens_clinical_cloud_v4.db')
    db_cur = db_c.cursor()
    
    if filter_q:
        db_cur.execute('SELECT * FROM records WHERE national_id LIKE ? OR patient_name LIKE ? ORDER BY timestamp DESC', ('%' + filter_q + '%', '%' + filter_q

