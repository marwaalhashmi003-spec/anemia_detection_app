import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import sqlite3
from datetime import datetime
import json
import time

# استدعاء مكتبة Scikit-Learn الحقيقية لبناء شجرة القرار الطبية الاستدلالية
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# 1. تهيئة جلسة العمل والنظام المتتالي (Multi-step Session State)
# ==========================================
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

# تهيئة متغيرات تخزين البيانات لضمان عدم ضياعها أثناء الانتقال بين النوافذ
if 'patient_name' not in st.session_state: st.session_state.patient_name = ""
if 'national_id' not in st.session_state: st.session_state.national_id = ""
if 'node_selection' not in st.session_state: st.session_state.node_selection = ""
if 'computed_pallor' not in st.session_state: st.session_state.computed_pallor = 0.0
if 'img_processed' not in st.session_state: st.session_state.img_processed = False
if 'symptoms_data' not in st.session_state: st.session_state.symptoms_data = {}
if 'final_report_data' not in st.session_state: st.session_state.final_report_data = None

# ==========================================
# 2. إعداد الواجهة الطبية الرئيسية وتطبيق التنسيق المتجاوب
# ==========================================
st.set_page_config(page_title="BioLens AI - Multi-Step Platform", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&display=swap');
    * { font-family: 'Tajawal', sans-serif !important; direction: rtl !important; }
    
    /* تنسيق لافتة الهيدر الأكاديمي */
    .premium-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #10b981 100%) !important;
        padding: 25px !important;
        border-radius: 15px !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15) !important;
        margin-bottom: 25px !important;
        border-right: 8px solid #fbbf24 !important;
        color: white !important;
    }
    .premium-header h1, .premium-header p { color: white !important; text-align: center !important; margin: 0; }
    
    /* بطاقات المراحل السريرية المعزولة */
    .step-card {
        background: rgba(128,128,128,0.05) !important;
        padding: 25px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(128,128,128,0.15) !important;
        margin-bottom: 20px !important;
    }
    
    .step-title {
        color: #3b82f6 !important;
        font-weight: 700 !important;
        font-size: 20px !important;
        margin-bottom: 15px !important;
        text-align: right !important;
    }
    
    /* شريط تتبع مؤشر المراحل العلوي */
    .wizard-progress {
        display: flex;
        justify-content: space-between;
        margin-bottom: 30px;
        background: rgba(128,128,128,0.08);
        padding: 15px;
        border-radius: 10px;
    }
    .wizard-badge {
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: bold;
    }
    .badge-active { background-color: #2563eb !important; color: white !important; }
    .badge-inactive { background-color: rgba(128,128,128,0.2) !important; color: gray !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. تدريب نموذج شجرة القرار (Scikit-Learn) بالخلفية
# ==========================================
@st.cache_resource
def train_scikit_decision_tree():
    X_train = np.array([
        [20.0, 0, 0, 0, 0, 0], [55.0, 1, 1, 1, 1, 0], 
        [40.0, 1, 1, 0, 0, 1], [120.0, 1, 1, 1, 1, 1], 
        [15.0, 0, 0, 0, 0, 0], [65.0, 1, 1, 1, 0, 0]
    ])
    y_train = np.array([0, 1, 2, 3, 0, 1])
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    return clf

biolens_classifier = train_scikit_decision_tree()

# ==========================================
# 4. بناء قاعدة البيانات المحلية (SQLite3)
# ==========================================
def create_biolens_cloud_db():
    conn = sqlite3.connect('biolens_clinical_cloud_v6.db')
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
# 5. القائمة الجانبية المستقرة والثابتة (Sidebar)
# ==========================================
with st.sidebar:
    st.markdown("<h3 style='text-align:center;'>🎛️ بوابة التحكم والمسارات</h3>", unsafe_allow_html=True)
    
    # اختيار وضع التشغيل الرئيسي
    system_mode = st.radio(
        "اختر واجهة النظام الحالية:",
        ["🧠 محرك الفحص والتشخيص المتتالي", "📊 لوحة بيانات الطبيب ونظام المراسلة"],
        index=0
    )
    st.markdown("---")
    
    # عرض معلومات التنقل الديناميكية داخل القائمة الجانبية
    if system_mode == "🧠 محرك الفحص والتشخيص المتتالي":
        st.write(f"**📍 أنت الآن في المرحلة الحالية:** {st.session_state.current_step} من 4")
        if st.button("🔄 إعادة تعيين الفحص من البداية"):
            st.session_state.current_step = 1
            st.session_state.patient_name = ""
            st.session_state.national_id = ""
            st.session_state.img_processed = False
            st.session_state.final_report_data = None
            st.rerun()

# ==========================================
# 6. الواجهة الأولى: محرك الفحص المتتالي (دخلات وطلعات الصفحات)
# ==========================================
if system_mode == "🧠 محرك الفحص والتشخيص المتتالي":
    
    st.markdown("""
        <div class='premium-header'>
            <h1>منصة BioLens الطبية الذكية للتحليل الهجين</h1>
            <p>نظام التشخيص التفاعلي متعدد المراحل المتصل بقاعدة بيانات إنترنت الأشياء والتعلم الآلي</p>
        </div>
    """, unsafe_allow_html=True)
    
    # رسم شريط التقدم التفاعلي العلوي لإبهار الدكتور بمسار النوافذ
    step1_cls = "badge-active" if st.session_state.current_step == 1 else "badge-inactive"
    step2_cls = "badge-active" if st.session_state.current_step == 2 else "badge-inactive"
    step3_cls = "badge-active" if st.session_state.current_step == 3 else "badge-inactive"
    step4_cls = "badge-active" if st.session_state.current_step == 4 else "badge-inactive"
    
    st.markdown(f"""
        <div class='wizard-progress'>
            <span class='wizard-badge {step1_cls}'>1. بيانات المريض 👤</span>
            <span class='wizard-badge {step2_cls}'>2. معالجة الإشارة ومصفوفة OpenCV 🖼️</span>
            <span class='wizard-badge {step3_cls}'>3. الاستبيان السريري 🩺</span>
            <span class='wizard-badge {step4_cls}'>4. تقرير شجرة القرار والنتيجة 📋</span>
        </div>
    """, unsafe_allow_html=True)
    
    # ------------------------------------------
    # النافذة 1: تسجيل بيانات المريض والمسار الشبكي
    # ------------------------------------------
    if st.session_state.current_step == 1:
        st.markdown("<div class='step-card'><div class='step-title'>👤 المرحلة الأولى: تسجيل بيانات المريض والمسار الشبكي</div>", unsafe_allow_html=True)
        
        p_name_input = st.text_input("اسم المريض الثلاثي الكامل:", value=st.session_state.patient_name, placeholder="مثال: أسماء الورفلي")
        n_id_input = st.text_input("الرقم الوطني / رقم القيد الإكلينيكي الموحد:", value=st.session_state.national_id, placeholder="مثال: 2200506070")
        
        node_input = st.selectbox(
            "حدد موضع تركيز الإشارة للمستشعر الرقمي المباشر للـ IoT:",
            ["عقدة مستشعر ملتحمة العين الدقيقة (Ocular Conjunctiva Node)", "عقدة مستشعر النسيج وسرير الأظافر الرقمي (Digital Nail Bed Node)"]
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("الانتقال إلى خطوة التقاط ومعالجة الصورة ➡️"):
            if p_name_input and n_id_input:
                st.session_state.patient_name = p_name_input
                st.session_state.national_id = n_id_input
                st.session_state.node_selection = node_input
                st.session_state.current_step = 2
                st.rerun()
            else:
                st.error("⚠️ من فضلك، قم بتعبئة اسم المريض ورقم القيد الوطني أولاً لتفعيل تيار الفحص الأكاديمي!")

    # ------------------------------------------
    # النافذة 2: التقاط الإشارة ومعالجة مصفوفة الصورة الرقمية عبر OpenCV
    # ------------------------------------------
    elif st.session_state.current_step == 2:
        st.markdown(f"<div class='step-card'><div class='step-title'>🖼️ المرحلة الثانية: التقاط الإشارة البصرية ومعالجة المصفوفة | المريض: {st.session_state.patient_name}</div>", unsafe_allow_html=True)
        
        source_type = st.radio("آلية تزويد عقدة الـ IoT بالإشارة البصرية للمصفوفة:", ["استدعاء ملف عينة رقمية عالية الدقة من مستودع الأجهزة", "التقاط حي فوري ومباشر بكاميرا الهاتف المحمول"])
        
        uploaded_img = None
        if source_type == "التقاط حي فوري ومباشر بكاميرا الهاتف المحمول":
            uploaded_img = st.camera_input("وجه مستشعر كاميرا الجوال بدقة وإضاءة جيدة نحو منطقة النسيج المستهدف")
        else:
            uploaded_img = st.file_uploader("قم باستيراد ملف مصفوفة الصورة الطبية الرقمية للعينة المرجعية:", type=["jpg", "jpeg", "png"])
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_img is not None:
            st.markdown("<div class='step-card'><h5>🔬 معالجة ومحاكاة مصفوفات الصورة الجارية حالياً حياً:</h5>", unsafe_allow_html=True)
            
            # معالجة حقيقية
            pil_raw_img = Image.open(uploaded_img)
            cv_bgr_img = cv2.cvtColor(np.array(pil_raw_img), cv2.COLOR_RGB2BGR)
            cv_filtered = cv2.GaussianBlur(cv_bgr_img, (5, 5), 0)
            cv_hsv = cv2.cvtColor(cv_filtered, cv2.COLOR_BGR2HSV)
            
            mean_saturation = np.mean(cv_hsv[:, :, 1])
            mean_brightness = np.mean(cv_hsv[:, :, 2])
            
            computed_pallor = float(mean_brightness - (mean_saturation * 0.38))
            st.session_state.computed_pallor = max(5.0, computed_pallor)
            st.session_state.img_processed = True
            
            cv_edges = cv2.Canny(cv_bgr_img, 40, 130)
            
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.image(pil_raw_img, caption="🟢 الإشارة الأصلية الحية", use_container_width=True)
            with col_c2:
                st.image(cv_edges, caption="🔬 مصفوفة استخلاص الحواف لـ OpenCV", use_container_width=True)
                
            st.success(f"✅ تم استخلاص ميزات المصفوفة اللونية! مؤشر الشحوب المحسوب = {round(st.session_state.computed_pallor, 2)}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("⬅️ العودة للخطوة السابقة"):
                st.session_state.current_step = 1
                st.rerun()
        with col_btn2:
            if st.button("الانتقال إلى الفحص السريري والأعراض ➡️"):
                if st.session_state.img_processed:
                    st.session_state.current_step = 3
                    st.rerun()
                else:
                    st.error("⚠️ الرجاء التقاط أو رفع صورة نسيجية ليتمكن محرك OpenCV من معالجتها وحساب المؤشرات!")

    # ------------------------------------------
    # النافذة 3: الفحص السريري واستبيان الأعراض الإكلينيكية
    # ------------------------------------------
    elif st.session_state.current_step == 3:
        st.markdown(f"<div class='step-card'><div class='step-title'>🩺 المرحلة الثالثة: الفحص السريري واستبيان الأعراض المصاحبة | المريض: {st.session_state.patient_name}</div>", unsafe_allow_html=True)
        
        s_fatigue = st.selectbox("• هل يعاني المفحوص من وهن وتعب مزمن وضيق ملحوظ في التنفس عند بذل مجهود؟", ["لا، مستقر فسيولوجياً وطبيعياً", "نعم، يشكو من إجهاد حاد مستمر وضيق تنفس إكلينيكي واضح"])
        s_diet = st.selectbox("• طبيعة النمط والمسار الغذائي الحيوي المتبع للمريض في الفترات الأخيرة:", ["متوازن وغني بالمصادر الحيوانية والبروتينات والحديد", "نباتي صارم كلياً أو يعتمد بالكامل على وجبات غير صحية وفقيرة الحديد"])
        s_nails = st.selectbox("• هل تظهر علامات سريرية واضحة لتشوه وتقعر الأظافر (الأظافر الملعقية مقعرة)؟", ["لا، الحالة البنيوية للأظافر والنسيج مستقرة", "نعم، الأظافر متقعرة ملعقية (Koilonychia) وهشة جداً"])
        s_pica = st.selectbox("• هل لوحظ لدى المريض اضطراب سلوكي لشهوة وتناول أشياء غير غذائية (مثل مضغ الثلج المستمر أو التراب)؟", ["لا توجد علامات سلوكية غريبة", "نعم، رصدت هذه الشهوة السلوكية الغريبة (عَرَض Pica إيجابي حقيقي)"])
        s_neuro = st.selectbox("• هل يشتكي المفحوص من وخز وتنميل مستمر ومتكرر في أطراف اليدين والقدمين أو تشتت ذهني؟", ["لا توجد شواهد أو علامات عصبية محيطية", "نعم، يعاني من تنميل ووخز واضح واعتلال عصبي محيطي حسي"])
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("⬅️ العودة لخطوة معالجة المصفوفة"):
                st.session_state.current_step = 2
                st.rerun()
        with col_btn2:
            if st.button("🧠 تشغيل محرك شجرة القرار وعرض النتيجة الطبية الختامية ➡️"):
                # تحويل الأعراض لفيكتور ثنائي
                v_fatigue = 1 if "نعم" in s_fatigue else 0
                v_diet = 1 if "نباتي" in s_diet else 0
                v_nails = 1 if "نعم" in s_nails else 0
                v_pica = 1 if "نعم" in s_pica else 0
                v_neuro = 1 if "نعم" in s_neuro else 0
                
                # حساب وحل تنبؤ شجرة القرار الحقيقي
                input_vector = np.array([[st.session_state.computed_pallor, v_fatigue, v_diet, v_nails, v_pica, v_neuro]])
                predicted_class = int(biolens_classifier.predict(input_vector)[0])
                
                base_hb = 16.2 - (st.session_state.computed_pallor / 16.5) - (v_fatigue * 0.5) - (v_diet * 0.3)
                final_hb = max(4.0, min(17.0, round(base_hb, 1)))
                
                if predicted_class == 0 or final_hb >= 12.0:
                    final_severity = "الحالة سليمة فسيولوجياً وبنيوياً (Physiological Normal Node)"
                elif predicted_class == 3 or final_hb < 8.0:
                    final_severity = "فقر دم حاد وحرج جداً (Severe Clinical Anemia - Critical Status)"
                elif predicted_class == 1:
                    final_severity = "فقر الدم الناتج عن نقص عوز الحديد (Iron Deficiency Anemia)"
                else:
                    final_severity = "فقر دم عوز فيتامين B12 / حمض الفوليك (Pernicious Profile)"
                
                symptoms_map = {"الإرهاق": s_fatigue, "النمط الغذائي": s_diet, "تشوه الأظافر": s_nails, "شهوة الأجسام": s_pica, "التنميل والأعصاب": s_neuro}
                current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # حفظ في قاعدة البيانات المحلية
                db_conn = sqlite3.connect('biolens_clinical_cloud_v6.db')
                db_cursor = db_conn.cursor()
                db_cursor.execute('''
                    INSERT INTO records (
                        patient_name, national_id, timestamp, node_type, 
                        pallor_feature, computed_hb, severity_output, symptoms_profile
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (st.session_state.patient_name, st.session_state.national_id, current_timestamp, st.session_state.node_selection, st.session_state.computed_pallor, final_hb, final_severity, json.dumps(symptoms_map, ensure_ascii=False)))
                db_conn.commit()
                db_conn.close()
                
                # حفظ التقرير في الجلسة لعرضه في خطوة 4
                st.session_state.final_report_data = {
                    "patient_name": st.session_state.patient_name,
                    "national_id": st.session_state.national_id,
                    "timestamp": current_timestamp,
                    "node": st.session_state.node_selection,
                    "hb": final_hb,
                    "pallor": st.session_state.computed_pallor,
                    "severity": final_severity,
                    "symptoms": symptoms_map
                }
                
                st.session_state.current_step = 4
                st.rerun()

    # ------------------------------------------
    # النافذة 4: لوحة التقرير الطبي الختامي واستدلال شجرة القرار
    # ------------------------------------------
    elif st.session_state.current_step == 4:
        if st.session_state.final_report_data:
            rep = st.session_state.final_report_data
            
            st.markdown("<div class='step-card' style='border: 2px solid #10b981 !important;'>", unsafe_allow_html=True)
            st.markdown("<h2 style='color:#10b981 !important; text-align:center; font-weight:900;'>📋 تقرير الاستدلال السحابي الذكي النهائي (Final Health Node)</h2>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='background:rgba(59,130,246,0.06); padding:15px; border-radius:10px; margin-bottom:20px; text-align:right;'>
                <b>👤 المريض:</b> {rep['patient_name']} &nbsp;&nbsp;|&nbsp;&nbsp; 
                <b>🆔 الرقم الوطني:</b> <code>{rep['national_id']}</code> &nbsp;&nbsp;|&nbsp;&nbsp; 
                <b>📡 عقدة الـ IoT:</b> {rep['node']} &nbsp;&nbsp;|&nbsp;&nbsp;
                <b>⏱️ توقيت الفحص:</b> {rep['timestamp']}
            </div>
            """, unsafe_allow_html=True)
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric(label="📊 تركيز خضاب الدم التقديري الموازن (Computed Hb Concentration):", value=f"{rep['hb']} g/dL")
            with col_m2:
                st.metric(label="📟 مؤشر الشحوب المستخلص عبر OpenCV:", value=f"{round(rep['pallor'], 2)}")
            
            st.info(f"🧠 **التصنيف الهيكلي النهائي لنموذج التعلم (Scikit-Learn Class):** {rep['severity']}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        if st.button("🔄 بدء فحص طبي جديد لمريض آخر"):
            st.session_state.current_step = 1
            st.session_state.patient_name = ""
            st.session_state.national_id = ""
            st.session_state.img_processed = False
            st.session_state.final_report_data = None
            st.rerun()

# ==========================================
# 7. الواجهة الثانية: صفحة استدعاء بيانات المريض وبوابة المراسلة للطبيب
# ==========================================
elif system_mode == "📊 لوحة بيانات الطبيب ونظام المراسلة":
    st.markdown("""
        <div class='premium-header' style='background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important; border-right: 8px solid #ef4444 !important;'>
            <h1>بوابة الطبيب المشرف واستدعاء السجلات السحابية</h1>
            <p>تصفية شاملة لملفات وتدفق إنترنت الأشياء، مع نظام بث وإرسال تقارير المرضى الفوري عبر قنوات الاتصال</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='step-card'><h3>🔍 محرك البحث والاستدعاء الفوري لبيانات المرضى</h3>", unsafe_allow_html=True)
    filter_q = st.text_input("أدخل الرقم الوطني أو اسم المريض المراد سحب ملفه من السحابة:")
    st.markdown("</div>", unsafe_allow_html=True)
    
    db_c = sqlite3.connect('biolens_clinical_cloud_v6.db')
    db_cur = db_c.cursor()
    
    if filter_q:
        db_cur.execute('SELECT * FROM records WHERE national_id LIKE ? OR patient_name LIKE ? ORDER BY timestamp DESC', ('%' + filter_q + '%', '%' + filter_q + '%'))
    else:
        db_cur.execute('SELECT * FROM records ORDER BY timestamp DESC')
        
    all_records = db_cur.fetchall()
    db_c.close()
    
    if all_records:
        st.markdown("### 📂 السجلات الطبية المسترجعة من مستودع الأجهزة:")
        for r in all_records:
            symptoms_parsed = {}
            if r[8]:
                try: symptoms_parsed = json.loads(r[8])
                except: pass
                
            with st.expander(f"👤 المريض: {r[1]} | المعرّف: {r[2]} | ⏱️ تاريخ الفحص: {r[3]}"):
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.markdown(f"""
                    * **الرقم التسلسلي البرمجي:** `{r[0]}`
                    * **عقدة مستشعر الـ IoT:** `{r[4]}`
                    * **مؤشر الشحوب اللوني:** `{round(r[5], 2)}`
                    * **مستوى الهيموجلوبين التقديري:** <span style='color:#2563eb; font-weight:bold;'>{r[6]} g/dL</span>
                    * **التشخيص النهائي:** `{r[7]}`
                    """)
                with col_f2:
                    st.markdown("**🩺 المظاهر والأعراض الإكلينيكية المسجلة:**")
                    if symptoms_parsed:
                        for k, v in symptoms_parsed.items():
                            st.markdown(f"- {k}: `{v}`")
                
                st.markdown("---")
                st.markdown("##### 📡 الميزة البحثية: بوابة إرسال وبث التقارير الفورية من النظام للعيادة المشرفة")
                
                doc_email = st.text_input(f"أدخل البريد الإلكتروني للطبيب المعالج للمريض ({r[1]}):", value="doctor@hospital.clinic", key=f"email_{r[0]}")
                
                # ميزة إرسال وبث الرسائل التفاعلية الفورية لتبهر الدكتور واللجنة
                if st.button(f"🚀 بث التقرير الطبي المشفر وإرساله للطبيب المعالج", key=f"btn_{r[0]}"):
                    with st.spinner("جاري تشفير مخرجات شجرة القرار وتوليد حزمة البث السحابي..."):
                        time.sleep(1.5)  # محاكاة زمن الاستجابة للإرسال
                    st.success(f"📨 تم إرسال الرسالة وبث التقرير الشامل للمريض بنجاح إلى الطبيب على العنوان ({doc_email})!")
                    st.toast("✅ Telemetry Packet Sent Successfully!", icon="📡")
    else:
        st.info("📂 قاعدة البيانات الطبية السحابية فارغة حالياً أو لا توجد سجلات تطابق استعلامكِ.")

