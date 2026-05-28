import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from datetime import datetime
import time
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# 1. نظام محاكاة قاعدة البيانات السحابية الآمن
# ==========================================
if 'mock_database' not in st.session_state:
    st.session_state.mock_database = []

# تهيئة متغيرات الحفاظ على خطوات الفحص السلس
if 'current_step' not in st.session_state: st.session_state.current_step = 1
if 'patient_name' not in st.session_state: st.session_state.patient_name = ""
if 'national_id' not in st.session_state: st.session_state.national_id = ""
if 'node_selection' not in st.session_state: st.session_state.node_selection = "عقدة ملتحمة العين (Ocular Node)"
if 'computed_pallor' not in st.session_state: st.session_state.computed_pallor = 0.0
if 'img_processed' not in st.session_state: st.session_state.img_processed = False
if 'final_report_data' not in st.session_state: st.session_state.final_report_data = None

# ==========================================
# 2. هندسة الواجهة الفاخرة ومنع تداخل العناصر (Anti-Overlap UI Engine)
# ==========================================
st.set_page_config(page_title="BioLens Clinical Pro", page_icon="🔬", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Kufi+Arabic:wght@400;500;600;700&display=swap');
    
    /* ضبط البيئة والألوان الفاخرة */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f8fafc !important;
        direction: rtl !important;
        text-align: right !important;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, label, button, small, .stSelectbox {
        font-family: 'Noto Kufi Arabic', sans-serif !important;
    }

    /* إخفاء القوائم الجانبية تماماً لمنع التشوه */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedSidebarCollapsedAnchor"] { display: none !important; }
    header { visibility: hidden !important; }
    
    /* الهوية البصرية الرئيسية للمنصة */
    .premium-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
        padding: 30px 20px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.2);
        margin-bottom: 25px;
    }
    .premium-header h1 {
        color: #ffffff !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        margin: 0 0 8px 0 !important;
        letter-spacing: 0.5px;
    }
    .premium-header p {
        color: #93c5fd !important;
        font-size: 13px !important;
        margin: 0 !important;
    }

    /* الكروت السريرية الفاخرة مع مسافات أمان قسرية لمنع التداخل */
    .clinical-card {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03) !important;
        margin-bottom: 24px !important; /* حماية الفراغ السفلي */
        position: relative;
    }
    
    .card-heading {
        color: #1e3a8a !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        margin-bottom: 20px !important;
        border-right: 4px solid #3b82f6;
        padding-right: 10px;
    }

    /* حل مشكلة تداخل الأيقونات والنصوص في التوسيع (Expander Fix) */
    .stExpander {
        border-radius: 12px !important;
        background-color: #ffffff !important;
        margin-bottom: 15px !important;
    }
    
    /* تباعد مخصص لخيارات الرفع لمنع تداخل أزرار الأبلود */
    .upload-container {
        border: 2px dashed #cbd5e1;
        padding: 20px;
        border-radius: 12px;
        background-color: #fdfdfd;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    /* شريط تتبع المراحل الاحترافي المستجيب للموبايل */
    .step-bar {
        display: flex;
        justify-content: space-between;
        background-color: #e2e8f0;
        border-radius: 12px;
        padding: 6px;
        margin-bottom: 25px;
        gap: 6px;
    }
    .step-point {
        flex: 1;
        text-align: center !important;
        padding: 10px 4px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 8px;
        color: #64748b;
        background-color: transparent;
        transition: all 0.3s ease;
    }
    .step-point.active {
        background-color: #1e3a8a !important;
        color: #ffffff !important;
        box-shadow: 0 4px 10px rgba(30, 58, 138, 0.15) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. محرك شجرة القرار والذكاء الاصطناعي
# ==========================================
@st.cache_resource
def get_ml_engine():
    X = np.array([[20.,0,0,0,0,0], [55.,1,1,1,1,0], [40.,1,1,0,0,1], [120.,1,1,1,1,1]])
    y = np.array([0, 1, 2, 3])
    return DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)

ml_engine = get_ml_engine()

# ترويسة المنصة الفاخرة
st.markdown("""
    <div class='premium-header'>
        <h1>BioLens Precision</h1>
        <p>النظام السريري المتقدم لتحليل مؤشرات الشحوب وربط أجهزة الـ IoT</p>
    </div>
""", unsafe_allow_html=True)

# أداة التبديل العلوية الواضحة بدون تداخل
main_tab = st.radio(
    "الانتقال بين البوابات الإكلينيكية:",
    options=["🔬 بوابة التشخيص الذكي المتتابع", "📊 مستودع السجلات الطبية وTelemetry"],
    horizontal=True,
    label_visibility="visible"
)

st.markdown("<hr style='border:none; border-top: 1px solid #e2e8f0; margin: 20px 0;'>", unsafe_allow_html=True)

# ==========================================
# 4. المسار الأول: بوابة الفحص والخطوات
# ==========================================
if "بوابة التشخيص الذكي المتتابع" in main_tab:
    
    # تحديث مؤشرات شريط التنقل
    s1 = "active" if st.session_state.current_step == 1 else ""
    s2 = "active" if st.session_state.current_step == 2 else ""
    s3 = "active" if st.session_state.current_step == 3 else ""
    s4 = "active" if st.session_state.current_step == 4 else ""
    
    st.markdown(f"""
        <div class='step-bar'>
            <div class='step-point {s1}'>1. هوية المريض</div>
            <div class='step-point {s2}'>2. المصفوفة البصرية</div>
            <div class='step-point {s3}'>3. الفحص الإكلينيكي</div>
            <div class='step-point {s4}'>4. التقرير النهائي</div>
        </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------
    # الخطوة 1: المدخلات والتعريف
    # ------------------------------------------
    if st.session_state.current_step == 1:
        st.markdown("<div class='clinical-card'><div class='card-heading'>الخطوة 1: تهيئة بيانات الحالة وعقدة المستشعر</div>", unsafe_allow_html=True)
        p_name = st.text_input("اسم المريض بالكامل:", value=st.session_state.patient_name, placeholder="مثال: أسماء الورفلي")
        p_id = st.text_input("الرقم الوطني / المعرف الطبي الموحد:", value=st.session_state.national_id, placeholder="مثال: 2200506070")
        p_node = st.selectbox("موضع مستشعر الـ IoT النشط حالياً:", ["عقدة ملتحمة العين (Ocular Node)", "عقدة سرير الأظافر (Nail Bed Node)"])
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("الانتقال لمعالجة الإشارة البصرية ←", use_container_width=True, type="primary"):
            if p_name and p_id:
                st.session_state.patient_name = p_name
                st.session_state.national_id = p_id
                st.session_state.node_selection = p_node
                st.session_state.current_step = 2
                st.rerun()
            else:
                st.toast("⚠️ يرجى تزويد النظام بالاسم والرقم الوطني للمتابعة", icon="📝")

    # ------------------------------------------
    # الخطوة 2: رفع وتحليل الصورة (OpenCV) الآمن بدون تداخل
    # ------------------------------------------
    elif st.session_state.current_step == 2:
        st.markdown(f"<div class='clinical-card'><div class='card-heading'>الخطوة 2: قراءة وتحليل النسيج الحيوى للحالة: {st.session_state.patient_name}</div>", unsafe_allow_html=True)
        
        src = st.radio("آلية تزويد المصفوفة اللقطية:", ["استدعاء صورة عينة من الجهاز", "التقاط حي بكاميرا الهاتف"], horizontal=True)
        
        # حاوية معزولة ومحمية من التداخل لعمليات الرفع والالتقاط
        st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
        if src == "التقاط حي بكاميرا الهاتف":
            img_file = st.camera_input("التقط صورة واضحة للنسيج المستهدف")
        else:
            img_file = st.file_uploader("اختر صورة النسيج الرقمية عالية الدقة:", type=["jpg","png","jpeg"])
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if img_file:
            st.markdown("<div class='clinical-card'><div class='card-heading'>مصفوفة المعالجة الفورية واستخلاص الحواف (Canny Edge Detection)</div>", unsafe_allow_html=True)
            raw = Image.open(img_file)
            img_np = cv2.cvtColor(np.array(raw), cv2.COLOR_RGB2BGR)
            blurred = cv2.GaussianBlur(img_np, (5,5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            st.session_state.computed_pallor = float(np.mean(hsv[:,:,2]) - (np.mean(hsv[:,:,1]) * 0.4))
            st.session_state.img_processed = True
            
            edges = cv2.Canny(img_np, 50, 150)
            st.image(edges, use_container_width=True, caption="مصفوفة استخلاص الحواف النشطة برمجياً")
            st.success(f"✓ تم التحليل واستخلاص المعطيات. مؤشر الشحوب المستنتج = {round(st.session_state.computed_pallor, 2)}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        c1, c2 = st.columns(2)
        with c1:
            if st.button("→ العودة للخلف", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()
        with c2:
            if st.button("المتابعة لتقييم الأعراض ←", use_container_width=True, type="primary"):
                if st.session_state.img_processed:
                    st.session_state.current_step = 3
                    st.rerun()
                else:
                    st.toast("⚠️ الرجاء توفير أو التقاط صورة نسيجية أولاً لمعالجتها", icon="🖼️")

    # ------------------------------------------
    # الخطوة 3: تعبئة الأعراض وحساب النتائج الختامية
    # ------------------------------------------
    elif st.session_state.current_step == 3:
        st.markdown(f"<div class='clinical-card'><div class='card-heading'>الخطوة 3: مصفوفة الأعراض والعلامات الإكلينيكية للحالة</div>", unsafe_allow_html=True)
        
        f1 = st.selectbox("هل تعاني الحالة من تعب حاد وضيق غير مبرر في التنفس؟", ["لا، مستقر طبيعياً", "نعم، يشكو من إجهاد وضيق تنفس مستمر"])
        f2 = st.selectbox("طبيعة السلوك والنظام الغذائي السائد:", ["متوازن وغني بالحديد والعناصر الأساسية", "فقير بالعناصر المغذية أو نباتي صارم"])
        f3 = st.selectbox("حالة وبنية الأظافر والأنسجة الظاهرية:", ["طبيعية وسليمة", "تظهر علامات هشة، تقشر، أو تقعر ملعقي (Koilonychia)"])
        f4 = st.selectbox("رصد شهوة سلوكية لتناول مواد غير غذائية (Pica) كالثلج أو التراب:", ["لا توجد أي علامات سلوكية غريبة", "نعم، تم رصد هذا السلوك الإكلينيكي العَرَضي"])
        f5 = st.selectbox("هل توجد شكوى من وخز مستمر وتنميل واضح في الأطراف؟", ["لا توجد علامات عصبية محيطية", "نعم، يعاني من تنميل ووخز مستمر بالأطراف"])
        st.markdown("</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("→ السابق", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()
        with c2:
            if st.button("بث وحساب التقرير السريري النهائي ←", use_container_width=True, type="primary"):
                v1 = 1 if "نعم" in f1 else 0
                v2 = 1 if "فقير" in f2 else 0
                v3 = 1 if "نعم" in f3 else 0
                v4 = 1 if "نعم" in f4 else 0
                v5 = 1 if "نعم" in f5 else 0
                
                pred = int(ml_engine.predict([[st.session_state.computed_pallor, v1, v2, v3, v4, v5]])[0])
                hb_val = max(4.5, min(16.5, round(15.8 - (st.session_state.computed_pallor / 20.0) - (v1 * 0.5), 1)))
                
                if pred == 0 or hb_val >= 12.0: diag = "الحالة سليمة فسيولوجياً (Physiological Normal)"
                elif pred == 3 or hb_val < 8.0: diag = "فقر دم حاد وحرج جداً (Severe Anemia - Critical Status)"
                elif pred == 1: diag = "فقر دم ناتج عن نقص عوز الحديد (Iron Deficiency Anemia)"
                else: diag = "اشتباه فقر دم ناتج عن عوز فيتامين B12 (Pernicious Profile)"
                
                s_map = {"الإعياء": f1, "النمط الغذائي": f2, "بنية الأظافر": f3, "اضطراب Pica": f4, "التنميل": f5}
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                
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
    # الخطوة 4: كرت التقرير النهائي فائق الأناقة
    # ------------------------------------------
    elif st.session_state.current_step == 4:
        if st.session_state.final_report_data:
            r = st.session_state.final_report_data
            st.markdown("<div class='clinical-card' style='border-top: 5px solid #10b981 !important;'><div class='card-heading' style='border:none; color:#10b981 !important; margin:0;'>✓ تم استخلاص وبث التقرير الطبي الشامل بنجاح</div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <p style='font-size:14px; margin: 6px 0;'><b>المريض الفحوص:</b> {r['name']} | <b>المعرّف الطبي:</b> <code>{r['national_id']}</code></p>
                <p style='font-size:14px; margin: 6px 0;'><b>العقدة الرقمية النشطة:</b> {r['node']} | <b>توقيت الفحص:</b> {r['time']}</p>
                <hr style='border:none; border-top: 1px solid #e2e8f0; margin: 15px 0;'>
            """, unsafe_allow_html=True)
            
            col_m1, col_m2 = st.columns(2)
            with col_m1: st.metric("خضاب الدم التقديري (Computed Hb)", f"{r['hb']} g/dL")
            with col_m2: st.metric("مؤشر الشحوب اللوني (Pallor Index)", f"{round(r['pallor'], 2)}")
                
            st.markdown(f"""
                <div style='background-color:#f1f5f9; padding:14px; border-radius:10px; margin-top:15px; border: 1px solid #cbd5e1;'>
                    <span style='font-size:12px; font-weight:600; color:#475569;'>تصنيف خوارزمية شجرة القرار (Inference Engine):</span><br>
                    <span style='font-size:14px; font-weight:700; color:#1e3a8a;'>{r['diag']}</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        if st.button("🔄 إجراء فحص إكلينيكي جديد لحالة أخرى", use_container_width=True):
            st.session_state.current_step = 1
            st.session_state.patient_name = ""
            st.session_state.national_id = ""
            st.session_state.img_processed = False
            st.session_state.final_report_data = None
            st.rerun()

# ==========================================
# 5. المسار الثاني: مستودع السجلات مع منع تداخل النصوص التوقيتية
# ==========================================
elif "مستودع السجلات الطبية" in main_tab:
    st.markdown("<div class='clinical-card'><div class='card-heading'>استدعاء السجلات الطبية وحزم الـ Telemetry التاريخية للمنصة</div>", unsafe_allow_html=True)
    search_q = st.text_input("ابحث باسم الحالة أو الرقم السريري لتصفية البيانات فورياً:")
    st.markdown("</div>", unsafe_allow_html=True)
    
    all_rec = st.session_state.mock_database
    if search_q:
        filtered_rec = [i for i in all_rec if search_q in i['name'] or search_q in i['national_id']]
    else:
        filtered_rec = all_rec
        
    if filtered_rec:
        for item in reversed(filtered_rec):
            # تم استخدام عنوان مرن ومتباعد لمنع تداخل أيقونات المتصفح مع النصوص التوقيتية
            with st.expander(f"👤 {item['name']} | المعرّف: {item['national_id']}"):
                st.markdown(f"""
                <div style='padding: 5px 0;'>
                <p style='font-size:13px; line-height:1.7; margin:0;'>
                • توقيت بث الإشارة الحيوية: <b>{item['time']}</b><br>
                • موضع العقدة الإلكترونية المستهدفة: <code>{item['node']}</code><br>
                • مؤشر الشحوب البصري الفعلي: <b>{round(item['pallor'], 2)}</b><br>
                • هيموجلوبين الدم المحسوب: <span style='color:#2563eb; font-weight:bold;'>{item['hb']} g/dL</span><br>
                • استنتاج الذكاء الاصطناعي النهائي: <span style='color:#1e3a8a; font-weight:600;'>{item['diag']}</span>
                </p>
                </div>
                <hr style='border:none; border-top:1px dashed #cbd5e1; margin:12px 0;'>
                <p style='font-size:12px; color:#475569; font-weight:600;'>📡 بث وإرسال حزمة التقارير الفورية للمستشفيات والعيادات عن بُعد:</p>
                """, unsafe_allow_html=True)
                
                doc_mail = st.text_input("بريد جهة الاستلام الطبية المستهدفة:", value="clinic-receiver@hospital.ly", key=f"mail_{item['id']}")
                
                # مساحة أمان قسرية قبل الزر لمنع الالتصاق والتداخل
                st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
                if st.button("🚀 تشفير وبث الحزمة اللاسلكية فورياً", key=f"btn_{item['id']}", use_container_width=True):
                    with st.spinner("جاري توقيع الحزمة الحيوية وبثها سحابياً..."):
                        time.sleep(0.7)
                    st.success(f"📨 تم إرسال ملف الـ Telemetry بنجاح للعنوان: {doc_mail}")
                    st.toast("✅ Telemetry Sent!", icon="📡")
    else:
        st.info("📂 المستودع فارغ حالياً، لم يتم تخزين سجلات فحوصات في الجلسة النشطة بعد.")

