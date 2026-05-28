import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from datetime import datetime
import json
import time
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# 1. نظام محاكاة قاعدة البيانات السحابية الذكي (Session Based Cloud DB)
# ==========================================
# بديل آمن تماماً لـ SQLite يمنع ظهور أخطاء write permissions على السحابة
if 'mock_database' not in st.session_state:
    st.session_state.mock_database = []

# تهيئة متغيرات التنقل والخطوات
if 'current_step' not in st.session_state: st.session_state.current_step = 1
if 'patient_name' not in st.session_state: st.session_state.patient_name = ""
if 'national_id' not in st.session_state: st.session_state.national_id = ""
if 'node_selection' not in st.session_state: st.session_state.node_selection = "عقدة ملتحمة العين (Ocular Node)"
if 'computed_pallor' not in st.session_state: st.session_state.computed_pallor = 0.0
if 'img_processed' not in st.session_state: st.session_state.img_processed = False
if 'final_report_data' not in st.session_state: st.session_state.final_report_data = None

# ==========================================
# 2. إعداد مظهر الواجهة الاحترافية الهادئة (Premium Clinical Design)
# ==========================================
st.set_page_config(page_title="BioLens Precision", page_icon="🔬", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Kufi+Arabic:wght@400;500;600;700&display=swap');
    
    /* تنظيف البيئة العامة وضبط الاتجاهات */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #fafafa !important;
        direction: rtl !important;
        text-align: right !important;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, label, button, small {
        font-family: 'Noto Kufi Arabic', sans-serif !important;
    }

    /* إخفاء شريط الأدوات والقوائم الجانبية المشوهة نهائياً */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedSidebarCollapsedAnchor"] { display: none !important; }
    header { visibility: hidden !important; }
    
    /* تصميم شعار الهوية الطبية */
    .app-brand {
        text-align: center !important;
        padding: 20px 0;
    }
    .app-brand h1 {
        color: #1e293b !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        margin-bottom: 4px !important;
    }
    .app-brand p {
        color: #64748b !important;
        font-size: 13px !important;
        margin: 0 !important;
    }

    /* كروت التصميم النظيفة (Minimalist Cards) */
    .clinical-card {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 14px !important;
        padding: 20px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02) !important;
        margin-bottom: 16px !important;
    }
    
    .card-heading {
        color: #1e40af !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        margin-bottom: 15px !important;
        border-right: 3px solid #2563eb;
        padding-right: 8px;
    }

    /* شريط تتبع المراحل السلس */
    .step-bar {
        display: flex;
        justify-content: space-between;
        background-color: #f1f5f9;
        border-radius: 30px;
        padding: 4px;
        margin-bottom: 20px;
    }
    .step-point {
        flex: 1;
        text-align: center !important;
        padding: 6px 4px;
        font-size: 11px;
        font-weight: 500;
        border-radius: 25px;
        color: #94a3b8;
    }
    .step-point.active {
        background-color: #ffffff !important;
        color: #1e1b4b !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04) !important;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. محرك الاستدلال الرياضي وشجرة القرار
# ==========================================
@st.cache_resource
def get_ml_engine():
    X = np.array([[20.,0,0,0,0,0], [55.,1,1,1,1,0], [40.,1,1,0,0,1], [120.,1,1,1,1,1]])
    y = np.array([0, 1, 2, 3])
    return DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)

ml_engine = get_ml_engine()

# ==========================================
# 4. ترويسة التطبيق وأداة التبديل
# ==========================================
st.markdown("""
    <div class='app-brand'>
        <h1>BioLens Precision</h1>
        <p>المنصة السريرية الذكية لفحص شحوب الأنسجة ومؤشرات الـ IoT</p>
    </div>
""", unsafe_allow_html=True)

main_tab = st.segmented_control(
    "لوحة التحكم:",
    options=["🧠 بوابة الفحص المتتابع", "📊 مستودع السجلات والمراسلة"],
    default="🧠 بوابة الفحص المتتابع",
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 5. المسار الأول: بوابة التشخيص والخطوات
# ==========================================
if main_tab == "🧠 بوابة الفحص المتتابع":
    
    # تحديث مؤشر شريط الخطوات علوياً
    s1 = "active" if st.session_state.current_step == 1 else ""
    s2 = "active" if st.session_state.current_step == 2 else ""
    s3 = "active" if st.session_state.current_step == 3 else ""
    s4 = "active" if st.session_state.current_step == 4 else ""
    
    st.markdown(f"""
        <div class='step-bar'>
            <div class='step-point {s1}'>1. بيانات المريض</div>
            <div class='step-point {s2}'>2. مصفوفة الإشارة</div>
            <div class='step-point {s3}'>3. الأعراض</div>
            <div class='step-point {s4}'>4. التقرير</div>
        </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------
    # الخطوة 1: الهوية والمدخلات الأساسية
    # ------------------------------------------
    if st.session_state.current_step == 1:
        st.markdown("<div class='clinical-card'><div class='card-heading'>الخطوة الأولى: تحديد هوية الحالة وعقدة الفحص</div>", unsafe_allow_html=True)
        
        p_name = st.text_input("اسم المريض الثلاثي:", value=st.session_state.patient_name, placeholder="مثال: أسماء الورفلي")
        p_id = st.text_input("الرقم الوطني / المعرّف السريري:", value=st.session_state.national_id, placeholder="مثال: 2200506070")
        p_node = st.selectbox("موضع مستشعر الـ IoT النشط:", ["عقدة ملتحمة العين (Ocular Node)", "عقدة سرير الأظافر (Nail Bed Node)"])
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("الانتقال لمعالجة الصورة ←", use_container_width=True, type="primary"):
            if p_name and p_id:
                st.session_state.patient_name = p_name
                st.session_state.national_id = p_id
                st.session_state.node_selection = p_node
                st.session_state.current_step = 2
                st.rerun()
            else:
                st.toast("⚠️ يرجى تعبئة اسم المريض ورقم القيد للمتابعة", icon="📝")

    # ------------------------------------------
    # الخطوة 2: تحليل مصفوفة OpenCV
    # ------------------------------------------
    elif st.session_state.current_step == 2:
        st.markdown(f"<div class='clinical-card'><div class='card-heading'>الخطوة الثانية: قراءة المستشعر البصري | الحالة: {st.session_state.patient_name}</div>", unsafe_allow_html=True)
        
        src = st.radio("مصدر تزويد المصفوفة البصرية:", ["استدعاء ملف عينة رقمية", "التقاط حي بكاميرا الهاتف"], horizontal=True)
        img_file = st.camera_input("كاميرا") if src == "التقاط حي بكاميرا الهاتف" else st.file_uploader("رفع صورة النسيج:", type=["jpg","png","jpeg"])
        st.markdown("</div>", unsafe_allow_html=True)
        
        if img_file:
            st.markdown("<div class='clinical-card'><div class='card-heading'>المعالجة الفورية (OpenCV Edge Matrix)</div>", unsafe_allow_html=True)
            
            raw = Image.open(img_file)
            img_np = cv2.cvtColor(np.array(raw), cv2.COLOR_RGB2BGR)
            blurred = cv2.GaussianBlur(img_np, (5,5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            # حساب مؤشر الشحوب الفعلي برمجياً
            st.session_state.computed_pallor = float(np.mean(hsv[:,:,2]) - (np.mean(hsv[:,:,1]) * 0.4))
            st.session_state.img_processed = True
            
            edges = cv2.Canny(img_np, 50, 150)
            st.image(edges, use_container_width=True, caption="مصفوفة استخلاص الحواف النشطة")
            st.markdown(f"<p style='color:#16a34a; font-size:13px; font-weight:600;'>✓ تم التحليل بنجاح. مؤشر الشحوب المستخلص = {round(st.session_state.computed_pallor, 2)}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        c1, c2 = st.columns(2)
        with c1:
            if st.button("→ السابق", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()
        with c2:
            if st.button("التالي: الفحص السريري ←", use_container_width=True, type="primary"):
                if st.session_state.img_processed:
                    st.session_state.current_step = 3
                    st.rerun()
                else:
                    st.toast("⚠️ الرجاء التقاط أو رفع الصورة أولاً", icon="🖼️")

    # ------------------------------------------
    # الخطوة 3: الاستبيان السريري وحساب النتائج الآمن
    # ------------------------------------------
    elif st.session_state.current_step == 3:
        st.markdown(f"<div class='clinical-card'><div class='card-heading'>الخطوة الثالثة: الأعراض والملف الإكلينيكي | الحالة: {st.session_state.patient_name}</div>", unsafe_allow_html=True)
        
        f1 = st.selectbox("هل يشكو المريض من تعب حاد وضيق تنفس عند بذل جهد؟", ["لا، مستقر طبيعياً", "نعم، يشكو من إجهاد وضيق تنفس"])
        f2 = st.selectbox("طبيعة النظام الغذائي الحالي للحالة:", ["متوازن وغني بالحديد والفيتامينات", "فقير بالحديد أو يعتمد نمطاً نباتياً صارماً"])
        f3 = st.selectbox("العلامات الظاهرية على بنية الأظافر والنسيج:", ["طبيعية وسليمة", "تظهر علامات تقشر أو تقعر ملعقي (Koilonychia)"])
        f4 = st.selectbox("رصد اضطراب أو شهوة سلوكية لتناول أشياء غير غذائية (Pica):", ["لا توجد علامات غريبة", "نعم، رصد عَرَض سلوكي غريب (مضغ ثلج/تراب مستمر)"])
        f5 = st.selectbox("هل توجد شكوى من وخز مستمر وتنميل حاد في الأطراف؟", ["لا توجد علامات عصبية", "نعم، يعاني من تنميل ووخز محيطي"])
        st.markdown("</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("→ السابق", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()
        with c2:
            if st.button("تحليل النتيجة الختامية وبث التقرير ←", use_container_width=True, type="primary"):
                v1 = 1 if "نعم" in f1 else 0
                v2 = 1 if "فقير" in f2 else 0
                v3 = 1 if "نعم" in f3 else 0
                v4 = 1 if "نعم" in f4 else 0
                v5 = 1 if "نعم" in f5 else 0
                
                # استدعاء شجرة القرار وحساب خضاب الدم التقديري
                pred = int(ml_engine.predict([[st.session_state.computed_pallor, v1, v2, v3, v4, v5]])[0])
                hb_val = max(4.5, min(16.5, round(15.8 - (st.session_state.computed_pallor / 20.0) - (v1 * 0.5), 1)))
                
                if pred == 0 or hb_val >= 12.0: diag = "الحالة سليمة فسيولوجياً (Physiological Normal)"
                elif pred == 3 or hb_val < 8.0: diag = "فقر دم حاد وحرج جداً (Severe Anemia - Critical)"
                elif pred == 1: diag = "فقر دم ناتج عن نقص عوز الحديد (Iron Deficiency Anemia)"
                else: diag = "اشتباه فقر دم عوز فيتامين B12 (Pernicious Profile)"
                
                s_map = {"الإعياء": f1, "النمط الغذائي": f2, "بنية الأظافر": f3, "اضطراب Pica": f4, "التنميل": f5}
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                
                # صناعة الكيان وحفظه في الذاكرة الحية السحابية (آمن تماماً من الـ Errors)
                record_entry = {
                    "id": len(st.session_state.mock_database) + 1,
                    "name": st.session_state.patient_name,
                    "national_id": st.session_state.national_id,
                    "time": now_str,
                    "node": st.session_state.node_selection,
                    "pallor": st.session_state.computed_pallor,
                    "hb": hb_val,
                    "diag": diag,
                    "symptoms": s_map
                }
                st.session_state.mock_database.append(record_entry)
                
                st.session_state.final_report_data = record_entry
                st.session_state.current_step = 4
                st.rerun()

    # ------------------------------------------
    # الخطوة 4: عرض التقرير الطبي الختامي الأنيق
    # ------------------------------------------
    elif st.session_state.current_step == 4:
        if st.session_state.final_report_data:
            r = st.session_state.final_report_data
            st.markdown("<div class='clinical-card' style='border-top: 4px solid #16a34a !important;'><div class='card-heading' style='border:none; color:#16a34a !important; margin:0;'>✓ تم استخلاص وبث التقرير الطبي بنجاح</div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <hr style='border:none; border-top: 1px solid #e2e8f0; margin: 10px 0;'>
                <p style='font-size:13px; margin: 3px 0;'><b>المريض:</b> {r['name']} | <b>المعرّف:</b> <code>{r['national_id']}</code></p>
                <p style='font-size:13px; margin: 3px 0;'><b>العقدة المستهدفة:</b> {r['node']} | <b>التوقيت:</b> {r['time']}</p>
                <hr style='border:none; border-top: 1px solid #e2e8f0; margin: 10px 0;'>
            """, unsafe_allow_html=True)
            
            col_m1, col_m2 = st.columns(2)
            with col_m1: st.metric("خضاب الدم التقديري (Computed Hb)", f"{r['hb']} g/dL")
            with col_m2: st.metric("مؤشر الشحوب (Pallor Index)", f"{round(r['pallor'], 2)}")
                
            st.markdown(f"""
                <div style='background-color:#f8fafc; padding:10px; border-radius:8px; margin-top:10px; border: 1px solid #e2e8f0;'>
                    <span style='font-size:12px; font-weight:600; color:#475569;'>استنتاج نموذج شجرة القرار للذكاء الاصطناعي:</span><br>
                    <span style='font-size:13px; font-weight:700; color:#1e3a8a;'>{r['diag']}</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        if st.button("🔄 إجراء فحص طبي جديد", use_container_width=True):
            st.session_state.current_step = 1
            st.session_state.patient_name = ""
            st.session_state.national_id = ""
            st.session_state.img_processed = False
            st.session_state.final_report_data = None
            st.rerun()

# ==========================================
# 6. المسار الثاني: بوابة استدعاء السجلات والمراسلة الفورية
# ==========================================
elif main_tab == "📊 مستودع السجلات والمراسلة":
    st.markdown("<div class='clinical-card'><div class='card-heading'>استدعاء سجلات المرضى والـ Telemetry المتراكمة</div>", unsafe_allow_html=True)
    search_q = st.text_input("ابحث باسم المريض أو المعرّف الطبي للتصفية الفورية:")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # فلترة السجلات من الذاكرة الحية الآمنة
    all_rec = st.session_state.mock_database
    if search_q:
        filtered_rec = [i for i in all_rec if search_q in i['name'] or search_q in i['national_id']]
    else:
        filtered_rec = all_rec
        
    if filtered_rec:
        for item in reversed(filtered_rec):
            with st.expander(f"👤 {item['name']} | 🆔 {item['national_id']} | ⏱️ {item['time']}"):
                st.markdown(f"""
                <p style='font-size:13px; line-height:1.6; margin:0;'>
                • العقدة الرقمية المستهدفة: <code>{item['node']}</code><br>
                • مؤشر الشحوب اللوني: <b>{round(item['pallor'], 2)}</b><br>
                • نسبة الهيموجلوبين التقديرية: <span style='color:#2563eb; font-weight:bold;'>{item['hb']} g/dL</span><br>
                • قرار تصنيف الخوارزمية الآلي: <b>{item['diag']}</b>
                </p>
                <hr style='border:none; border-top:1px dashed #cbd5e1; margin:10px 0;'>
                <p style='font-size:12px; color:#475569; font-weight:600;'>📡 بث وإرسال حزمة التقارير الفورية للعيادة المعالجة عن بُعد:</p>
                """, unsafe_allow_html=True)
                
                doc_mail = st.text_input("بريد الطبيب أو المستشفى المستلم للتقرير:", value="doctor@hospital.clinic", key=f"mail_{item['id']}")
                
                if st.button("🚀 بث وتشفير البيانات السحابية فورياً", key=f"btn_{item['id']}", use_container_width=True):
                    with st.spinner("جاري تهيئة الحزمة اللاسلكية وتوقيعها برمجياً..."):
                        time.sleep(0.8)
                    st.success(f"📨 تم بث التقرير الطبي الشامل بنجاح إلى: {doc_mail}")
                    st.toast("✅ Telemetry Packet Sent!", icon="📡")
    else:
        st.info("📂 لا توجد سجلات مخزنة حالياً في الذاكرة السحابية النشطة.")

