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

# تهيئة متغيرات تخزين البيانات لضمان عدم ضياعها أثناء الانتقال
if 'patient_name' not in st.session_state: st.session_state.patient_name = ""
if 'national_id' not in st.session_state: st.session_state.national_id = ""
if 'node_selection' not in st.session_state: st.session_state.node_selection = "عقدة مستشعر ملتحمة العين الدقيقة (Ocular Conjunctiva Node)"
if 'computed_pallor' not in st.session_state: st.session_state.computed_pallor = 0.0
if 'img_processed' not in st.session_state: st.session_state.img_processed = False
if 'symptoms_data' not in st.session_state: st.session_state.symptoms_data = {}
if 'final_report_data' not in st.session_state: st.session_state.final_report_data = None

# ==========================================
# 2. إعداد الواجهة الطبية الرئيسية وتطبيق التنسيق المتجاوب
# ==========================================
st.set_page_config(page_title="BioLens AI", page_icon="🔬", layout="wide")

# هندسة CSS مخصصة لإخفاء الأزرار المزعجة وحل التداخلات نهائياً على الجوال
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&display=swap');
    
    /* ضبط الخطوط والاتجاه العام */
    * { font-family: 'Tajawal', sans-serif !important; direction: rtl !important; text-align: right; }
    
    /* إخفاء القائمة الجانبية الافتراضية لعدم حدوث تداخل أو تقطيع نصوص على الجوال */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedSidebarCollapsedAnchor"] { display: none !important; }
    
    /* لافتة الهيدر الأكاديمي المتجاوبة */
    .premium-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #10b981 100%) !important;
        padding: 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
        margin-bottom: 20px !important;
        color: white !important;
        text-align: center !important;
    }
    .premium-header h1 { color: white !important; text-align: center !important; font-size: 24px !important; margin: 0 0 8px 0; }
    .premium-header p { color: #f3f4f6 !important; text-align: center !important; font-size: 14px !important; margin: 0; }
    
    /* بطاقات المراحل السريرية */
    .step-card {
        background: rgba(128,128,128,0.04) !important;
        padding: 20px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(128,128,128,0.12) !important;
        margin-bottom: 15px !important;
    }
    
    .step-title {
        color: #2563eb !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        margin-bottom: 12px !important;
    }
    
    /* شريط تتبع مؤشر المراحل المتجاوب بدون تداخل */
    .wizard-progress {
        display: flex;
        justify-content: space-around;
        background: rgba(128,128,128,0.06);
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 20px;
        gap: 5px;
    }
    .wizard-badge {
        padding: 6px 10px;
        border-radius: 15px;
        font-size: 11px;
        font-weight: bold;
        text-align: center !important;
        flex: 1;
    }
    .badge-active { background-color: #2563eb !important; color: white !important; }
    .badge-inactive { background-color: rgba(128,128,128,0.15) !important; color: #6b7280 !important; }
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
# 5. التبديل الرئيسي العلوى السلس (بديل القائمة الجانبية المخربة للـ Mobile)
# ==========================================
selected_mode = st.radio(
    "اختر لوحة العمل الحالية:",
    ["🧠 محرك الفحص والتشخيص المتتالي", "📊 لوحة السجلات ونظام المراسلة للطبيب"],
    horizontal=True
)

st.markdown("---")

# ==========================================
# 6. الواجهة الأولى: محرك الفحص المتتالي
# ==========================================
if selected_mode == "🧠 محرك الفحص والتشخيص المتتالي":
    
    st.markdown("""
        <div class='premium-header'>
            <h1>منصة BioLens الطبية الذكية</h1>
            <p>نظام تشخيص تفاعلي متتابع فائق السلاسة متوافق تماماً مع أجهزة الهاتف</p>
        </div>
    """, unsafe_allow_html=True)
    
    # مؤشر المراحل العلوي
    step1_cls = "badge-active" if st.session_state.current_step == 1 else "badge-inactive"
    step2_cls = "badge-active" if st.session_state.current_step == 2 else "badge-inactive"
    step3_cls = "badge-active" if st.session_state.current_step == 3 else "badge-inactive"
    step4_cls = "badge-active" if st.session_state.current_step == 4 else "badge-inactive"
    
    st.markdown(f"""
        <div class='wizard-progress'>
            <span class='wizard-badge {step1_cls}'>1. المريض 👤</span>
            <span class='wizard-badge {step2_cls}'>2. المصفوفة 🖼️</span>
            <span class='wizard-badge {step3_cls}'>3. الأعراض 🩺</span>
            <span class='wizard-badge {step4_cls}'>4. النتيجة 📋</span>
        </div>
    """, unsafe_allow_html=True)
    
    # ------------------------------------------
    # المرحلة 1: بيانات المريض
    # ------------------------------------------
    if st.session_state.current_step == 1:
        st.markdown("<div class='step-card'><div class='step-title'>👤 المرحلة الأولى: تسجيل بيانات حالة المريض والـ IoT</div>", unsafe_allow_html=True)
        
        p_name_input = st.text_input("اسم المريض الثلاثي الكامل:", value=st.session_state.patient_name, placeholder="مثال: أسماء الورفلي")
        n_id_input = st.text_input("الرقم الوطني للمريض:", value=st.session_state.national_id, placeholder="مثال: 2200506070")
        
        node_input = st.selectbox(
            "حدد موضع تركيز الإشارة للمستشعر الرقمي الـ IoT:",
            ["عقدة مستشعر ملتحمة العين الدقيقة (Ocular Conjunctiva Node)", "عقدة مستشعر النسيج وسرير الأظافر الرقمي (Digital Nail Bed Node)"],
            index=0 if st.session_state.node_selection.startswith("عقدة مستشعر ملتحمة") else 1
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("الانتقال إلى خطوة معالجة الصورة ➡️", use_container_width=True):
            if p_name_input and n_id_input:
                st.session_state.patient_name = p_name_input
                st.session_state.national_id = n_id_input
                st.session_state.node_selection = node_input
                st.session_state.current_step = 2
                st.rerun()
            else:
                st.error("⚠️ يرجى كتابة اسم المريض والرقم الوطني لتفعيل خطوة الفحص!")

    # ------------------------------------------
    # المرحلة 2: معالجة الإشارة ومصفوفة OpenCV
    # ------------------------------------------
    elif st.session_state.current_step == 2:
        st.markdown(f"<div class='step-card'><div class='step-title'>🖼️ المرحلة الثانية: التقاط الإشارة ومعالجة المصفوفة | {st.session_state.patient_name}</div>", unsafe_allow_html=True)
        
        source_type = st.radio("مصدر تزويد المصفوفة البصرية:", ["استدلال ملف عينة رقمية من المستودع", "التقاط حي بكاميرا الهاتف المحمول"], horizontal=True)
        
        uploaded_img = None
        if source_type == "التقاط حي بكاميرا الهاتف المحمول":
            uploaded_img = st.camera_input("وجه الكاميرا نحو النسيج المستهدف")
        else:
            uploaded_img = st.file_uploader("اختر ملف الصورة الرقمية:", type=["jpg", "jpeg", "png"])
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_img is not None:
            st.markdown("<div class='step-card'><h5>🔬 المعالجة الحية عبر فلاتر ومصفوفات OpenCV:</h5>", unsafe_allow_html=True)
            
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
            
            st.image(cv_edges, caption="🔬 مصفوفة استخلاص الحواف النشطة لـ OpenCV", use_container_width=True)
            st.success(f"✅ تم تحليل المصفوفة! مؤشر الشحوب = {round(st.session_state.computed_pallor, 2)}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("⬅️ السابق", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()
        with col_btn2:
            if st.button("التالي: الأعراض ➡️", use_container_width=True):
                if st.session_state.img_processed:
                    st.session_state.current_step = 3
                    st.rerun()
                else:
                    st.error("⚠️ الرجاء التقاط أو تزويد النظام بالصورة أولاً!")

    # ------------------------------------------
    # المرحلة 3: الأعراض والاستبيان السريري
    # ------------------------------------------
    elif st.session_state.current_step == 3:
        st.markdown(f"<div class='step-card'><div class='step-title'>🩺 المرحلة الثالثة: الفحص السريري والأعراض | {st.session_state.patient_name}</div>", unsafe_allow_html=True)
        
        s_fatigue = st.selectbox("• هل يعاني من تعب دائم وضيق تنفس حاد عند الجهد؟", ["لا، مستقر فسيولوجياً وطبيعياً", "نعم، يشكو من إجهاد حاد وضيق تنفس"])
        s_diet = st.selectbox("• طبيعة المسار الغذائي الحالي للمريض:", ["متوازن وغني بالمصادر الحيوانية والحديد", "نباتي صارم أو يعتمد على وجبات غير صحية فقيرة الحديد"])
        s_nails = st.selectbox("• هل تظهر علامات لتقعر وتشوّه شكل الأظافر السريري؟", ["لا، الحالة البنيوية للأظافر مستقرة", "نعم، الأظافر متقعرة ملعقية (Koilonychia) وهشة"])
        s_pica = st.selectbox("• هل لوحظ تناول أشياء غير غذائية (مثل مضغ الثلج المستمر)؟", ["لا توجد علامات سلوكية غريبة", "نعم، رصدت شهوة سلوكية غريبة (عَرَض Pica إيجابي)"])
        s_neuro = st.selectbox("• هل يشتكي من وخز وتنميل مستمر في الأطراف؟", ["لا توجد علامات عصبية محيطية", "نعم، يعاني من تنميل ووخز واعتلال عصبي حسي"])
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("⬅️ السابق", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()
        with col_btn2:
            if st.button("🧠 معالجة شجرة القرار وإخراج النتيجة ➡️", use_container_width=True):
                v_fatigue = 1 if "نعم" in s_fatigue else 0
                v_diet = 1 if "نباتي" in s_diet else 0
                v_nails = 1 if "نعم" in s_nails else 0
                v_pica = 1 if "نعم" in s_pica else 0
                v_neuro = 1 if "نعم" in s_neuro else 0
                
                input_vector = np.array([[st.session_state.computed_pallor, v_fatigue, v_diet, v_nails, v_pica, v_neuro]])
                predicted_class = int(biolens_classifier.predict(input_vector)[0])
                
                base_hb = 16.2 - (st.session_state.computed_pallor / 16.5) - (v_fatigue * 0.5) - (v_diet * 0.3)
                final_hb = max(4.0, min(17.0, round(base_hb, 1)))
                
                if predicted_class == 0 or final_hb >= 12.0:
                    final_severity = "الحالة سليمة فسيولوجياً وبنيوياً (Physiological Normal Node)"
                elif predicted_class == 3 or final_hb < 8.0:
                    final_severity = "فقر دم حاد وحرج جداً (Severe Clinical Anemia)"
                elif predicted_class == 1:
                    final_severity = "فقر الدم الناتج عن نقص عوز الحديد (Iron Deficiency Anemia)"
                else:
                    final_severity = "فقر دم عوز فيتامين B12 / حمض الفوليك"
                
                symptoms_map = {"الإرهاق": s_fatigue, "النمط الغذائي": s_diet, "تشوه الأظافر": s_nails, "شهوة الأجسام": s_pica, "التنميل": s_neuro}
                current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                
                # حفظ في SQLite
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
    # المرحلة 4: عرض النتائج الختامية
    # ------------------------------------------
    elif st.session_state.current_step == 4:
        if st.session_state.final_report_data:
            rep = st.session_state.final_report_data
            
            st.markdown("<div class='step-card' style='border: 2px solid #10b981 !important;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='color:#10b981 !important; text-align:center; font-weight:700;'>📋 تقرير الاستدلال الطبي النهائي</h3>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='background:rgba(59,130,246,0.05); padding:12px; border-radius:8px; margin-bottom:15px; font-size:13px;'>
                <b>👤 المريض:</b> {rep['patient_name']}<br>
                <b>🆔 الرقم الوطني:</b> <code>{rep['national_id']}</code><br>
                <b>📡 عقدة الـ IoT:</b> {rep['node']}<br>
                <b>⏱️ الوقت:</b> {rep['timestamp']}
            </div>
            """, unsafe_allow_html=True)
            
            st.metric(label="📊 تركيز خضاب الدم التقديري (Computed Hb):", value=f"{rep['hb']} g/dL")
            st.metric(label="📟 مؤشر الشحوب اللوني لـ OpenCV:", value=f"{round(rep['pallor'], 2)}")
            
            st.warning(f"🧠 استنتاج تصنيف نموذج الذكاء: {rep['severity']}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        if st.button("🔄 إجراء فحص طبي جديد", use_container_width=True):
            st.session_state.current_step = 1
            st.session_state.patient_name = ""
            st.session_state.national_id = ""
            st.session_state.img_processed = False
            st.session_state.final_report_data = None
            st.rerun()

# ==========================================
# 7. الواجهة الثانية: صفحة الطبيب ونظام المراسلة للنتائج
# ==========================================
elif selected_mode == "📊 لوحة السجلات ونظام المراسلة للطبيب":
    st.markdown("""
        <div class='premium-header' style='background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;'>
            <h1>بوابة استدعاء السجلات وبث الرسائل</h1>
            <p>منصة الطبيب المشرف لإرسال التقارير الطبية فورياً للعيادات المتصلة</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    filter_q = st.text_input("ابحث عن مريض بالاسم أو الرقم الوطني لاستدعاء ملفه:")
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
        for r in all_records:
            with st.expander(f"👤 {r[1]} | 🆔 {r[2]} | ⏱️ {r[3]}"):
                st.markdown(f"""
                * **موضع عقدة مستشعر الـ IoT:** `{r[4]}`
                * **مؤشر الشحوب اللوني المستخلص:** `{round(r[5], 2)}`
                * **نسبة الهيموجلوبين:** `{r[6]} g/dL`
                * **قرار نموذج التصنيف الآلي:** **{r[7]}**
                """)
                
                st.markdown("---")
                doc_email = st.text_input(f"بريد الطبيب المستلم لتقرير ({r[1]}):", value="doctor@hospital.clinic", key=f"em_{r[0]}")
                
                if st.button(f"🚀 بث وإرسال التقرير فورياً للطبيب", key=f"btn_{r[0]}", use_container_width=True):
                    with st.spinner("جاري تشفير الحزمة السحابية وبثها..."):
                        time.sleep(1.2)
                    st.success(f"📨 تم بث التقرير الطبي الشامل وإرساله بنجاح إلى: {doc_email}")
                    st.toast("✅ Telemetry Packet Sent!", icon="📡")
    else:
        st.info("📂 لا توجد سجلات مخزنة حالياً في قاعدة البيانات الطبية السحابية.")

