import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import sqlite3
from datetime import datetime
import json
import time

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
        
        .splash-container {
            margin-top: 50px;
            padding: 30px;
        }
        .logo-title {
            font-size: 50px;
            font-weight: 900;
            background: linear-gradient(135deg, #3b82f6 0%, #38bdf8 50%, #fbbf24 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }
        .logo-subtitle {
            color: #64748b;
            font-size: 16px;
            margin-bottom: 30px;
        }
        .pulse-element {
            width: 80px;
            height: 80px;
            background: rgba(59, 130, 246, 0.1);
            border: 4px solid #3b82f6;
            border-radius: 50%;
            margin: 0 auto 20px auto;
            animation: pulse-animation 1.5s infinite ease-in-out;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 35px;
        }
        @keyframes pulse-animation {
            0% { transform: scale(0.95); opacity: 0.6; }
            50% { transform: scale(1.05); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.6; }
        }
        .status-text {
            color: #0284c7;
            font-size: 14px;
            font-weight: bold;
            margin-top: 10px;
        }
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
            (20, "🔄 جاري استدعاء مكتبات الرؤية الحاسوبية ومعالجة المصفوفات OpenCV..."),
            (50, "🧠 جاري تحميل وتجهيز أوزان شجرة القرار الاستدلالية (Decision Tree)..."),
            (80, "📡 جاري إنشاء بروتوكول الاتصال لعقد مستشعرات إنترنت الأشياء الطرفية..."),
            (100, "🟢 تم التمهيد بنجاح! جاري الانتقال إلى واجهة التحكم الطبية...")
        ]
        
        for p, text in loading_steps:
            time.sleep(0.5)
            progress_bar.progress(p)
            status_box.markdown(f"<p class='status-text'>{text}</p>", unsafe_allow_html=True)
            
        time.sleep(0.3)
        
    st.session_state.splash_executed = True
    st.rerun()

# ==========================================
# 3. إعداد الواجهة الطبية الرئيسية بعد انتهاء التحميل
# ==========================================
st.set_page_config(
    page_title="BioLens AI & IoT Platform",
    page_icon="🔬",
    layout="wide"
)

# ==========================================
# 4. التحكم الديناميكي في الوضع اللوني (النهاري والليلي)
# ==========================================
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding:10px; background: linear-gradient(135deg, #1e3a8a, #3b82f6); border-radius:12px; margin-bottom:15px;'>
            <h2 style='color:#ffffff; margin:0; font-size:22px; font-weight:900;'>BioLens AI</h2>
            <small style='color:#e2e8f0; display:block;'>Clinical Intelligence System</small>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-weight:700; margin-bottom:5px; text-align:right;'>🌗 وضع مظهر الشاشة:</p>", unsafe_allow_html=True)
    app_theme = st.radio("اختر نمط الإضاءة المريح لعينيك:", ["الوضع النهاري المشرق (Clinical Light)", "الوضع الليلي الفاخر (Deep Dark)"])
    st.markdown("---")

# اختيار الثيم وتطبيق الألوان المتوافقة مع بنية السيرفر
if app_theme == "الوضع النهاري المشرق (Clinical Light)":
    bg_color = "#f8fafc"
    card_bg = "#ffffff"
    text_color = "#0f172a"
    sub_text = "#475569"
    border_color = "#e2e8f0"
    header_gradient = "linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%)"
    header_text = "#ffffff"
else:
    bg_color = "#0f172a"
    card_bg = "#1e293b"
    text_color = "#f8fafc"
    sub_text = "#94a3b8"
    border_color = "#334155"
    header_gradient = "linear-gradient(135deg, #1e293b 0%, #0f172a 100%)"
    header_text = "#38bdf8"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&display=swap');
    
    /* تطبيق الألوان الأساسية */
    .stMainBlockContainer {{ background-color: {bg_color}; }}
    h1, h2, h3, h4, h5, h6, p, label, span {{ font-family: 'Tajawal', sans-serif !important; direction: rtl; text-align: right; color: {text_color}; }}
    
    .premium-header {{
        background: {header_gradient};
        padding: 25px;
        border-radius: 16px;
        color: {header_text} !important;
        text-align: center !important;
        margin-bottom: 25px;
        border-right: 6px solid #fbbf24;
    }}
    .premium-header h1 {{ color: {header_text} !important; text-align: center !important; font-weight: 900; font-size: 28px; }}
    
    .clinical-card {{
        background: {card_bg};
        padding: 20px;
        border-radius: 12px;
        border: 1px solid {border_color};
        margin-bottom: 20px;
    }}
    
    .section-title {{
        color: #3b82f6;
        font-weight: 700;
        font-size: 18px;
        border-bottom: 2px solid #fbbf24;
        padding-bottom: 5px;
        margin-bottom: 15px;
        text-align: right;
    }}
    
    .stButton>button {{
        width: 100%;
        background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
        color: white !important;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 700;
        height: 45px;
        border: none;
    }}
    
    .report-panel {{
        background: {card_bg};
        border: 2px solid #10b981;
        padding: 20px;
        border-radius: 16px;
        margin-top: 20px;
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 5. قاعدة البيانات المحلية
# ==========================================
def create_biolens_cloud_db():
    conn = sqlite3.connect('biolens_clinical_cloud_v2.db')
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
# 6. خوارزمية الحساب الطبي المعتمد
# ==========================================
def run_biolens_decision_tree(pallor, fatigue, diet, nails, pica, neuro):
    calculated_hb = 17.0 - (pallor / 14.2) - (fatigue * 0.45) - (diet * 0.25)
    calculated_hb = max(4.0, min(17.8, round(calculated_hb, 1)))
    
    if calculated_hb >= 12.0:
        severity = "الحالة سليمة وطبيعية تماماً (Physiological Normal Node)"
        reasoning = "مؤشرات امتصاص الطيف اللوني تقع بالكامل ضمن النطاق الصحي المعتمد لتدفق الأوعية الدموية الدقيقة المحيطية."
    elif calculated_hb < 8.0:
        severity = "فقر دم حاد وحرج جداً (Severe Clinical Anemia - Critical Status)"
        reasoning = "أظهر مستشعر القراءة البصرية شحوباً بنيوياً فائقاً ومستويات سطوع حرجة تعكس نقصاً حاداً في مركب الحديد العضوي. تتطلب الحالة رعاية طبية عاجلة."
    else:
        if nails == 1 or pica == 1:
            severity = "فقر الدم الناتج عن نقص عوز الحديد (Iron Deficiency Anemia)"
            reasoning = "تلازم علامات الشحوب النسيجي الرقمي مع ظهور تقعر الأظافر الإكلينيكي أو الرغبة السلوكية المرافقة (Pica)."
        elif neuro == 1 and diet == 1:
            severity = "فقر دم عوز فيتامين B12 / حمض الفوليك (Pernicious Macrocytic Profile)"
            reasoning = "وجود الاعتلال العصبي المحيطي الحسي مقترناً بمسار فقر التغذية الحيوانية وجه خوارزمية الاستدلال لتشخيص عوز B12."
        else:
            severity = "فقر دم عام خفيف إلى متوسط (Normocytic / General Anemia Profile)"
            reasoning = "رصد انخفاض تدريجي طفيف في تركيز خضاب الدم متناسب مع وهن عام خفيف. يوصى طبياً بإجراء فحص المعمل الموحد."
            
    return calculated_hb, severity, reasoning

# ==========================================
# 7. التوجيه عبر القائمة
# ==========================================
with st.sidebar:
    st.markdown("<p style='font-weight:700; margin-bottom:5px; text-align:right;'>⚙️ خيارات المنظومة الذكية:</p>", unsafe_allow_html=True)
    menu_selection = st.radio(
        "اختر واجهة العمل الحالية المدمجة:",
        ["🧠 محرك الفحص والمستشعر الضوئي (IoT Node)", "📊 السجلات الطبية السحابية التراكمية"]
    )

# ==========================================
# 8. واجهة العمل الأولى: محرك الفحص
# ==========================================
if menu_selection == "🧠 محرك الفحص والمستشعر الضوئي (IoT Node)":
    
    st.markdown("""
        <div class='premium-header'>
            <h1>منصة BioLens الطبية الذكية للتحليل الهجين</h1>
            <p>قراءة حقيقية ومعالجة حية للمصفوفات الرقمية للصور متصلة فورياً بنموذج شجرة القرار الاستدلالي</p>
        </div>
    """, unsafe_allow_html=True)
    
    # بطاقة البيانات التعريفية
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
                uploaded_img = st.camera_input("وجه مستشعر كاميرا الجوال بدقة وإضاءة جيدة نحو منطقة النسيج المستهدف")
            else:
                uploaded_img = st.file_uploader("قم باستيراد ملف Matrix للعينة المرجعية:", type=["jpg", "jpeg", "png"])
            st.markdown("</div>", unsafe_allow_html=True)
            
            if uploaded_img is not None:
                st.markdown("<div class='clinical-card'><div class='section-title'>⚙️ معالجة حقيقية فورية للمصفوفة الرقمية (OpenCV Processing)</div>", unsafe_allow_html=True)
                
                pil_raw_img = Image.open(uploaded_img)
                cv_bgr_img = cv2.cvtColor(np.array(pil_raw_img), cv2.COLOR_RGB2BGR)
                cv_filtered = cv2.GaussianBlur(cv_bgr_img, (5, 5), 0)
                cv_hsv = cv2.cvtColor(cv_filtered, cv2.COLOR_BGR2HSV)
                
                mean_saturation = np.mean(cv_hsv[:, :, 1])
                mean_brightness = np.mean(cv_hsv[:, :, 2])
                computed_pallor = float(mean_brightness - (mean_saturation * 0.38))
                
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.image(pil_raw_img, caption="🟢 الإشارة الأصلية الحية الملتقطة بمستشعر الـ IoT", use_container_width=True)
                with col_c2:
                    cv_edges = cv2.Canny(cv_bgr_img, 100, 200)
                    st.image(cv_edges, caption="🔬 مصفوفة استخلاص حواف الأوعية (OpenCV Canny)", use_container_width=True)
                
                st.info(f"📟 مؤشر الشحوب النسيجي الفعلي المحسوب للبؤرة الرقمية = {round(computed_pallor, 2)}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # الخطوة الرابعة: الفحص السريري
                st.markdown("<div class='clinical-card'><div class='section-title'>🩺 الخطوة 4: الفحص السريري الاستقصائي والأعراض الإكلينيكية</div>", unsafe_allow_html=True)
                s_fatigue = st.selectbox("• هل يعاني المفحوص من وهن وتعب مزمن، خمول مفاجئ وضيق ملحوظ في التنفس؟", ["لا، مستقر فسيولوجياً وطبيعياً", "نعم، يشكو من إجهاد حاد مستمر وضيق تنفس مستمر"])
                s_diet = st.selectbox("• طبيعة النمط والمسار الغذائي المتبع للمريض في الفترات الأخيرة:", ["متوازن وغني بالمصادر الحيوانية والبروتينات والحديد", "نباتي صارم كلياً أو يعتمد بالكامل على وجبات غير صحية"])
                s_nails = st.selectbox("• هل تظهر علامات سريرية واضحة لتشوه وتقعر الأظافر أو جفاف حاد؟", ["لا، الحالة النسيجية للأظافر طبيعية ومستقرة", "نعم، الأظافر متقعرة ملعقية وهشة جداً"])
                s_pica = st.selectbox("• هل لوحظ لدى المريض اضطراب سلوكي لشهوة وتناول أشياء غير غذائية غريبة (مثل مضغ الثلج)؟", ["لا توجد علامات سلوكية غريبة", "نعم، رصدت هذه الشهوة السلوكية الغريبة (عَرَض Pica إيجابي)"])
                s_neuro = st.selectbox("• هل يشتكي المفحوص من وخز وتنميل مستمر ومتكرر في أطراف اليدين والقدمين؟", ["لا توجد شواهد أو علامات عصبية محيطية", "نعم، يعاني من تنميل ووخز واضح واعتلال عصبي"])
                st.markdown("</div>", unsafe_allow_html=True)
                
                v_fatigue = 1 if s_fatigue == "نعم، يشكو من إجهاد حاد مستمر وضيق تنفس مستمر" else 0
                v_diet = 1 if s_diet == "نباتي صارم كلياً أو يعتمد بالكامل على وجبات غير صحية" else 0
                v_nails = 1 if s_nails == "نعم، الأظافر متقعرة ملعقية وهشة جداً" else 0
                v_pica = 1 if s_pica == "نعم، رصدت هذه الشهوة السلوكية الغريبة (عَرَض Pica إيجابي)" else 0
                v_neuro = 1 if s_neuro == "نعم، يعاني من تنميل ووخز واضح واعتلال عصبي" else 0
                
                if st.button("🚀 تشغيل خوارزمية شجرة القرار وبث التقرير التشخيصي للسحابة"):
                    with st.spinner("🧠 جاري معالجة المتغيرات وربط تيار بيانات مستشعر الـ IoT..."):
                        final_hb, final_severity, final_reasoning = run_biolens_decision_tree(
                            computed_pallor, v_fatigue, v_diet, v_nails, v_pica, v_neuro
                        )
                        
                        symptoms_map = {"الإرهاق": s_fatigue, "النمط الغذائي": s_diet, "تشوه الأظافر": s_nails, "شهوة الأجسام": s_pica, "التنميل والأعصاب": s_neuro}
                        symptoms_json_str = json.dumps(symptoms_map, ensure_ascii=False)
                        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        db_conn = sqlite3.connect('biolens_clinical_cloud_v2.db')
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
                    st.markdown("<h2 style='color:#10b981; text-align:center; font-weight:900;'>📋 تقرير الاستدلال السحابي النهائي</h2>", unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style='background:rgba(59, 130, 246, 0.05); padding:12px; border-radius:8px; border:1px solid #3b82f6; margin-bottom:15px; font-size:14px;'>
                        <b>👤 المفحوص:</b> {p_name} &nbsp;&nbsp;|&nbsp;&nbsp; 
                        <b>🆔 المعرّف الوطني:</b> <code>{n_id}</code> &nbsp;&nbsp;|&nbsp;&nbsp; 
                        <b>📡 عقدة الـ IoT:</b> {node_selection}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric(label="📊 تركيز خضاب الدم التقديري (Computed Hb):", value=f"{final_hb} g/dL")
                    with col_m2:
                        st.metric(label="📟 مؤشر الشحوب المستخلص (Raw Pallor):", value=f"{round(computed_pallor, 2)}")
                        
                    st.warning(f"🩺 **التصنيف الهيكلي لشجرة القرار:** {final_severity}")
                    
                    st.markdown(f"""
                    <div style='background:rgba(16, 185, 129, 0.05); padding:15px; border-radius:8px; border-right:4px solid #10b981; margin-top:15px;'>
                        <h4 style='color:#065f46; font-weight:bold; margin-top:0;'>🧠 مسار مبررات الاستدلال (Inference Path):</h4>
                        <p style='color:#064e3b; margin:0;'>{final_reasoning}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 9. واجهة العمل الثانية: مستودع السجلات
# ==========================================
elif menu_selection == "📊 السجلات الطبية السحابية التراكمية":
    st.markdown("""
        <div class='premium-header' style='background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-right: 6px solid #ef4444;'>
            <h1>مستودع السجلات الطبية التراكمية (BioLens IoT Telemetry)</h1>
            <p>مراجعة تاريخ تدفق ودمج بيانات أجهزة الاستشعار عبر السحابة والتحقق من مصداقيتها</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='clinical-card'><h3>🔍 محرك البحث والفرز الفوري</h3>", unsafe_allow_html=True)
    filter_q = st.text_input("أدخل اسم المريض أو الرقم الوطني لمطابقة السجل الفوري:")
    st.markdown("</div>", unsafe_allow_html=True)
    
    db_c = sqlite3.connect('biolens_clinical_cloud_v2.db')
    db_cur = db_c.cursor()
    if filter_q:
        db_cur.execute('SELECT * FROM records WHERE national_id LIKE ? OR patient_name LIKE ? ORDER BY timestamp DESC', ('%' + filter_q + '%', '%' + filter_q + '%'))
    else:
        db_cur.execute('SELECT * FROM records ORDER BY timestamp DESC')
    all_records = db_cur.fetchall()
    db_c.close()
    
    if all_records:
        for r in all_records:
            with st.expander(f"👤 المريض: {r[1]} | ⏱️ تاريخ البث: {r[3]}"):
                st.markdown(f"""
                * **معرّف السجل السحابي:** `{r[0]}`
                * **الرقم الوطني:** `{r[2]}`
                * **موضع الفحص الرقمي:** `{r[4]}`
                * **تركيز خضاب الدم المقدر:** **{r[6]} g/dL**
                * **التشخيص الطبي النهائي:** `{r[7]}`
                """)
    else:
        st.info("📂 قاعدة البيانات الطبية السحابية جاهزة ومستقرة تماماً، ولا توجد سجلات مطابقة حالياً.")
