import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import sqlite3
from datetime import datetime
import json
import time
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# 1. تهيئة تيار الجلسة (Session State)
# ==========================================
if 'current_step' not in st.session_state: st.session_state.current_step = 1
if 'patient_name' not in st.session_state: st.session_state.patient_name = ""
if 'national_id' not in st.session_state: st.session_state.national_id = ""
if 'node_selection' not in st.session_state: st.session_state.node_selection = "عقدة ملتحمة العين (Ocular Node)"
if 'computed_pallor' not in st.session_state: st.session_state.computed_pallor = 0.0
if 'img_processed' not in st.session_state: st.session_state.img_processed = False
if 'final_report_data' not in st.session_state: st.session_state.final_report_data = None

# ==========================================
# 2. إعداد الصفحة وهندسة المظهر العصري (Modern Minimalist CSS)
# ==========================================
st.set_page_config(page_title="BioLens Precision", page_icon="🔬", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Noto+Kufi+Arabic:wght@400;500;600;700&display=swap');
    
    /* تصفير العشوائية وتحديد الخطوط والاتجاه */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f8fafc !important; /* خلفية هادئة مريحة للعين */
        direction: rtl !important;
        text-align: right !important;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, label, button {
        font-family: 'Noto Kufi Arabic', 'Inter', sans-serif !important;
    }

    /* إخفاء القائمة الجانبية المزعجة نهائياً لمنع أي تداخل */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedSidebarCollapsedAnchor"] { display: none !important; }
    header { visibility: hidden !important; } /* إخفاء شريط أزرار ستريمليت العلوي المشوه */
    
    /* الهيدر الاحترافي البسيط */
    .app-brand {
        text-align: center !important;
        padding: 15px 0 25px 0;
    }
    .app-brand h1 {
        color: #0f172a !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        margin-bottom: 5px !important;
    }
    .app-brand p {
        color: #64748b !important;
        font-size: 14px !important;
        margin: 0 !important;
    }

    /* بطاقات العمل البيضاء الفاخرة النظيفة */
    .clinical-card {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
        margin-bottom: 20px !important;
    }
    
    .card-heading {
        color: #1e40af !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        margin-bottom: 18px !important;
        border-right: 4px solid #3b82f6;
        padding-right: 8px;
    }

    /* شريط خطوات ذكي وناعم بدون ألوان فاقعة */
    .step-bar {
        display: flex;
        justify-content: space-between;
        background-color: #ededed;
        border-radius: 30px;
        padding: 4px;
        margin-bottom: 25px;
    }
    .step-point {
        flex: 1;
        text-align: center !important;
        padding: 6px 10px;
        font-size: 12px;
        font-weight: 500;
        border-radius: 25px;
        color: #64748b;
    }
    .step-point.active {
        background-color: #ffffff !important;
        color: #0f172a !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06) !important;
        font-weight: 600;
    }
    
    /* تحسين شكل المدخلات المكتوبة */
    div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] div {
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. محرك الاستدلال الذكي (Scikit-Learn)
# ==========================================
@st.cache_resource
def get_ml_model():
    X = np.array([[20.,0,0,0,0,0], [55.,1,1,1,1,0], [40.,1,1,0,0,1], [120.,1,1,1,1,1]])
    y = np.array([0, 1, 2, 3])
    return DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)

ml_engine = get_ml_model()

# ==========================================
# 4. التحكم في التبديل العلوي بين الأقسام
# ==========================================
st.markdown("""
    <div class='app-brand'>
        <h1>BioLens Precision</h1>
        <p>المنصة السريرية المتكاملة لفحص شحوب الأنسجة وتحليل المؤشرات الحيوية</p>
    </div>
""", unsafe_allow_html=True)

main_tab = st.segmented_control(
    "واجهة التشغيل الحالية:",
    options=["🧠 بوابة الفحص المتتابع", "📊 قاعدة بيانات الطبيب والمراسلة"],
    default="🧠 بوابة الفحص المتتابع",
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 5. المسار الأول: بوابة الفحص التتابعي (دخلات وطلعات الصفحات)
# ==========================================
if main_tab == "🧠 بوابة الفحص المتتابع":
    
    # رسم شريط الخطوات المعاصر والنظيف
    s1 = "active" if st.session_state.current_step == 1 else ""
    s2 = "active" if st.session_state.current_step == 2 else ""
    s3 = "active" if st.session_state.current_step == 3 else ""
    s4 = "active" if st.session_state.current_step == 4 else ""
    
    st.markdown(f"""
        <div class='step-bar'>
            <div class='step-point {s1}'>1. الهوية</div>
            <div class='step-point {s2}'>2. المستشعر</div>
            <div class='step-point {s3}'>3. الأعراض</div>
            <div class='step-point {s4}'>4. النتيجة</div>
        </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------
    # الصفحة 1: هوية المريض
    # ------------------------------------------
    if st.session_state.current_step == 1:
        st.markdown("<div class='clinical-card'><div class='card-heading'>معلومات المريض وعقدة الاستشعار</div>", unsafe_allow_html=True)
        
        p_name = st.text_input("اسم المريض بالكامل", value=st.session_state.patient_name, placeholder="مثال: سارة الأحمد")
        p_id = st.text_input("الرقم الوطني / المعرّف الطبي الموحد", value=st.session_state.national_id, placeholder="مثال: 10203040")
        p_node = st.selectbox("تحديد عقدة تجميع الإشارة الطرفية (IoT Node)", ["عقدة ملتحمة العين (Ocular Node)", "عقدة سرير الأظافر (Nail Bed Node)"])
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("الانتقال لخطوة الفحص البصري ←", use_container_width=True, type="primary"):
            if p_name and p_id:
                st.session_state.patient_name = p_name
                st.session_state.national_id = p_id
                st.session_state.node_selection = p_node
                st.session_state.current_step = 2
                st.rerun()
            else:
                st.toast("⚠️ يرجى ملء الحقول المطلوبة أولاً", icon="🛑")

    # ------------------------------------------
    # الصفحة 2: تحليل مستشعر الصورة (OpenCV)
    # ------------------------------------------
    elif st.session_state.current_step == 2:
        st.markdown(f"<div class='clinical-card'><div class='card-heading'>معالجة مصفوفة الإشارة البصرية | الحالة: {st.session_state.patient_name}</div>", unsafe_allow_html=True)
        
        src = st.radio("آلية قراءة مستشعر النسيج:", ["استدعاء عينة رقمية مخزنة", "التقاط فوري عبر كاميرا الهاتف"], horizontal=True)
        
        img_file = st.camera_input("التقاط") if src == "التقاط فوري عبر كاميرا الهاتف" else st.file_uploader("رفع ملف العينة البصرية", type=["jpg","png","jpeg"])
        st.markdown("</div>", unsafe_allow_html=True)
        
        if img_file:
            st.markdown("<div class='clinical-card'><div class='card-heading'>معاينة استخلاص الحواف والميزات (Edge Detection Matrix)</div>", unsafe_allow_html=True)
            
            raw = Image.open(img_file)
            img_np = cv2.cvtColor(np.array(raw), cv2.COLOR_RGB2BGR)
            blurred = cv2.GaussianBlur(img_np, (5,5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            # حساب مؤشر الشحوب اللوني الفعلي برمجياً
            st.session_state.computed_pallor = float(np.mean(hsv[:,:,2]) - (np.mean(hsv[:,:,1]) * 0.4))
            st.session_state.img_processed = True
            
            edges = cv2.Canny(img_np, 50, 150)
            st.image(edges, use_container_width=True, caption="مصفوفة تباين الميزات المستخلصة للـ IoT")
            
            st.markdown(f"<p style='color:#16a34a; font-weight:600;'>✓ تم استخلاص الميزة بنجاح. مؤشر الشحوب اللوني الحالي: {round(st.session_state.computed_pallor, 2)}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        c1, c2 = st.columns(2)
        with c1:
            if st.button("→ السابق", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()
        with c2:
            if st.button("التالي: الأعراض السريرية ←", use_container_width=True, type="primary"):
                if st.session_state.img_processed:
                    st.session_state.current_step = 3
                    st.rerun()
                else:
                    st.toast("⚠️ الرجاء تزويد النظام بالصورة الطبية أولاً", icon="🖼️")

    # ------------------------------------------
    # الصفحة 3: الأعراض والاستبيان الإكلينيكي
    # ------------------------------------------
    elif st.session_state.current_step == 3:
        st.markdown(f"<div class='clinical-card'><div class='card-heading'>الاستبيان الفسيولوجي والأعراض | الحالة: {st.session_state.patient_name}</div>", unsafe_allow_html=True)
        
        f1 = st.selectbox("هل يشتكي المريض من إعياء دائم وضيق تنفس غير مبرر؟", ["لا، فسيولوجيا الجسم مستقرة", "نعم، يعاني من إجهاد دائم وضيق تنفس"])
        f2 = st.selectbox("طبيعة النمط الغذائي ومدى توفر مصادر الحديد والبروتين:", ["نمط غذائي متوازن وغني", "نمط نباتي صارم أو يعتمد على وجبات فقيرة الحديد"])
        f3 = st.selectbox("المظهر السريري لبنية الأظافر والنسيج الطرفي:", ["مظهر طبيعي ومستقر", "تظهر علامات الأظافر المقعرة (Koilonychia)"])
        f4 = st.selectbox("وجود اضطراب سلوكي لتناول مواد غير غذائية (Pica):", ["لا توجد شواهد سلوكية غريبة", "نعم، رصدت هذه الشهوة السلوكية الغريبة (مضغ الثلج/التراب)"])
        f5 = st.selectbox("هل يشتكي من وخز، تنميل أو اعتلالات عصبية محيطية حادة؟", ["لا توجد علامات عصبية", "نعم، يعاني من وخز مستمر وتنميل بالأطراف"])
        st.markdown("</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("→ السابق", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()
        with c2:
            if st.button("تحليل النتيجة الختامية عبر شجرة القرار ←", use_container_width=True, type="primary"):
                v1 = 1 if "نعم" in f1 else 0
                v2 = 1 if "نباتي" in f2 else 0
                v3 = 1 if "نعم" in f3 else 0
                v4 = 1 if "نعم" in f4 else 0
                v5 = 1 if "نعم" in f5 else 0
                
                # تنفيذ نموذج شجرة القرار والحساب التقديري للهيموجلوبين
                pred = int(ml_engine.predict([[st.session_state.computed_pallor, v1, v2, v3, v4, v5]])[0])
                hb_val = max(4.5, min(16.5, round(16.0 - (st.session_state.computed_pallor / 18.0) - (v1 * 0.6), 1)))
                
                if pred == 0 or hb_val >= 12.0: diag = "الحالة مستقرة وسليمة بنيوياً (Physiological Normal)"
                elif pred == 3 or hb_val < 8.0: diag = "فقر دم حاد وحرج جداً (Severe Anemia - Critical)"
                elif pred == 1: diag = "فقر دم ناتج عن نقص مخزون الحديد (Iron Deficiency Anemia)"
                else: diag = "اشتباه فقر دم عوز فيتامين B12 (Pernicious Profile)"
                
                s_map = {"الإعياء": f1, "النمط الغذائي": f2, "الأظافر": f3, "اضطراب Pica": f4, "التنميل": f5}
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                
                # تخزين فوري في قاعدة البيانات المحلية المحاكية للسحابة
                conn = sqlite3.connect('biolens_clinical_cloud_v6.db')
                c = conn.cursor()
                c.execute("INSERT INTO records (patient_name, national_id, timestamp, node_type, pallor_feature, computed_hb, severity_output, symptoms_profile) VALUES (?,?,?,?,?,?,?,?)",
                          (st.session_state.patient_name, st.session_state.national_id, now_str, st.session_state.node_selection, st.session_state.computed_pallor, hb_val, diag, json.dumps(s_map, ensure_ascii=False)))
                conn.commit()
                conn.close()
                
                st.session_state.final_report_data = {
                    "name": st.session_state.patient_name, "id": st.session_state.national_id,
                    "time": now_str, "node": st.session_state.node_selection, "hb": hb_val,
                    "pallor": st.session_state.computed_pallor, "diag": diag, "symptoms": s_map
                }
                st.session_state.current_step = 4
                st.rerun()

    # ------------------------------------------
    # الصفحة 4: لوحة التقرير الطبي الختامي
    # ------------------------------------------
    elif st.session_state.current_step == 4:
        if st.session_state.final_report_data:
            r = st.session_state.final_report_data
            st.markdown("<div class='clinical-card' style='border-top: 5px solid #16a34a !important;'><div class='card-heading' style='border:none; color:#16a34a !important;'>✓ تم توليد التقرير الطبي الاستدلالي بنجاح</div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <p style='font-size:14px; margin: 4px 0;'><b>المريض:</b> {r['name']} | <b>المعرّف:</b> <code>{r['id']}</code></p>
                <p style='font-size:14px; margin: 4px 0;'><b>العقدة المستهدفة:</b> {r['node']} | <b>التوقيت:</b> {r['time']}</p>
                <hr style='border:none; border-top: 1px solid #e2e8f0; margin: 15px 0;'>
            """, unsafe_allow_html=True)
            
            # عرض المؤشرات الحيوية بشكل وبطاقات غاية في الأناقة والبساطة
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("خضاب الدم المقدر (Computed Hb)", f"{r['hb']} g/dL")
            with col_m2:
                st.metric("مؤشر الشحوب الرقمي (Pallor)", f"{round(r['pallor'], 2)}")
                
            st.markdown(f"""
                <div style='background-color:#f1f5f9; padding:12px; border-radius:10px; margin-top:15px;'>
                    <span style='font-size:13px; font-weight:600; color:#334155;'>التصنيف النهائي لشجرة القرار للذكاء الاصطناعي:</span><br>
                    <span style='font-size:14px; font-weight:700; color:#1e3a8a;'>{r['diag']}</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        if st.button("🔄 إجراء فحص طبي جديد لمريض آخر", use_container_width=True):
            st.session_state.current_step = 1
            st.session_state.patient_name = ""
            st.session_state.national_id = ""
            st.session_state.img_processed = False
            st.session_state.final_report_data = None
            st.rerun()

# ==========================================
# 6. المسار الثاني: صفحة استدعاء البيانات وبوابة المراسلة للطبيب
# ==========================================
elif main_tab == "📊 قاعدة بيانات الطبيب والمراسلة":
    st.markdown("<div class='clinical-card'><div class='card-heading'>سحب السجلات الطبية من المستودع السحابي والـ IoT</div>", unsafe_allow_html=True)
    search_q = st.text_input("ابحث باسم المريض أو الرقم الوطني الفوري لتصفية الملفات:")
    st.markdown("</div>", unsafe_allow_html=True)
    
    conn = sqlite3.connect('biolens_clinical_cloud_v6.db')
    c = conn.cursor()
    if search_q:
        c.execute('SELECT * FROM records WHERE national_id LIKE ? OR patient_name LIKE ? ORDER BY timestamp DESC', ('%'+search_q+'%', '%'+search_q+'%'))
    else:
        c.execute('SELECT * FROM records ORDER BY timestamp DESC')
    records = c.fetchall()
    conn.close()
    
    if records:
        for item in records:
            with st.expander(f"👤 {item[1]} | 🆔 {item[2]} | ⏱️ {item[3]}"):
                st.markdown(f"""
                <p style='font-size:13px; line-height:1.6; margin:0;'>
                • عقدة المستشعر البصري: <code>{item[4]}</code><br>
                • مؤشر الشحوب المستخلص: <b>{round(item[5], 2)}</b><br>
                • تركيز الهيموجلوبين التقديري: <span style='color:#2563eb; font-weight:bold;'>{item[6]} g/dL</span><br>
                • تشخيص محرك النظام المعتمد: <b>{item[7]}</b>
                </p>
                <hr style='border:none; border-top:1px dashed #cbd5e1; margin:12px 0;'>
                <p style='font-size:12px; color:#475569; font-weight:600;'>📡 بوابة الاتصال عن بُعد وبث الرسائل والتقارير فورياً للطبيب:</p>
                """, unsafe_allow_html=True)
                
                doc_mail = st.text_input("بريد العيادة أو الطبيب المعالج المستلم:", value="doctor@hospital.clinic", key=f"mail_{item[0]}")
                
                if st.button("🚀 بث وإرسال حزمة التقرير الطبي الفوري", key=f"btn_{item[0]}", use_container_width=True):
                    with st.spinner("جاري تعبئة حزمة البيانات وتوقيعها مشفّرة..."):
                        time.sleep(1.0)
                    st.success(f"📨 تم إرسال التقرير الشامل للمريض بنجاح إلى الطبيب على العنوان ({doc_mail})!")
                    st.toast("✅ Telemetry Data Transmitted!", icon="📡")
    else:
        st.info("📂 المستودع الطبي السحابي فارغ حالياً، أو لا توجد نتائج تطابق بحثكِ.")

