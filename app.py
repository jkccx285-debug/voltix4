# -*- coding: utf-8 -*-
"""
الکترونیک‌یار
اپلیکیشن آموزشی فارسی برای هنرجویان الکترونیک

اجرا:
    streamlit run app.py

AI:
    کلید Gemini را در Streamlit Secrets با نام GEMINI_API_KEY قرار دهید.

نکته:
    رابط کاربری با HTML/CSS داخل همین فایل ساخته شده است، اما موتور اجرا
    Streamlit/Python است تا کلید API در سمت سرور باقی بماند.
"""

import io
import math
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import faiss
    FAISS_OK = True
except Exception:
    faiss = None
    FAISS_OK = False

try:
    from pypdf import PdfReader
    PYPDF_OK = True
except Exception:
    PdfReader = None
    PYPDF_OK = False

try:
    from google import genai
    from google.genai import types
    GEMINI_OK = True
except Exception:
    genai = None
    types = None
    GEMINI_OK = False


# ============================================================
# تنظیمات
# ============================================================

st.set_page_config(
    page_title="الکترونیک‌یار",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# ظاهر — HTML/CSS داخل همین فایل
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg: #f6f8fc;
    --card: #ffffff;
    --text: #172033;
    --muted: #667085;
    --line: #e5e7eb;
    --primary: #2563eb;
    --primary-soft: #eff6ff;
    --success: #047857;
    --success-soft: #ecfdf5;
    --warning: #b45309;
    --warning-soft: #fffbeb;
}

html, body, [class*="css"], button, input, textarea, select {
    font-family: "Vazirmatn", Tahoma, sans-serif !important;
}

.stApp {
    direction: rtl;
    background:
        radial-gradient(circle at 5% 0%, rgba(37,99,235,.06), transparent 25%),
        radial-gradient(circle at 95% 0%, rgba(16,185,129,.05), transparent 25%),
        var(--bg);
}

.block-container {
    max-width: 1240px;
    padding: 1.2rem .8rem 3rem;
}

.hero {
    background: linear-gradient(135deg,#111827,#1e293b);
    color: #fff;
    border-radius: 24px;
    padding: 28px 26px;
    margin-bottom: 20px;
    box-shadow: 0 14px 40px rgba(15,23,42,.16);
}

.hero h1 {
    margin: 0 0 8px;
    font-size: clamp(1.55rem,4vw,2.3rem);
    font-weight: 800;
}

.hero p {
    margin: 0;
    color: #dbeafe;
    line-height: 1.9;
}

.card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 18px;
    margin: 8px 0 16px;
    box-shadow: 0 5px 18px rgba(15,23,42,.045);
}

.formula {
    direction:ltr;
    text-align:center;
    font-family:monospace;
    background:#f8fafc;
    border:1px dashed #cbd5e1;
    border-radius:12px;
    padding:13px;
    margin:12px 0;
    font-size:1rem;
}

.result {
    background:var(--primary-soft);
    border:1px solid #bfdbfe;
    border-radius:16px;
    text-align:center;
    padding:18px;
    margin:12px 0;
}

.result .value {
    color:#1d4ed8;
    font-size:clamp(1.35rem,4vw,2rem);
    font-weight:800;
}

.result .label {
    color:var(--muted);
    margin-top:4px;
}

.circuit {
    direction:ltr;
    text-align:center;
    background:#fff;
    border:1px solid #cbd5e1;
    border-radius:16px;
    padding:18px;
    font-family:monospace;
    overflow-x:auto;
    margin:10px 0;
}

.note {
    background:#f8fafc;
    border-right:4px solid #94a3b8;
    padding:12px 15px;
    border-radius:10px;
    line-height:1.9;
}

.ok {
    background:var(--success-soft);
    border-right:4px solid #10b981;
    padding:12px 15px;
    border-radius:10px;
}

.warn {
    background:var(--warning-soft);
    border-right:4px solid #f59e0b;
    padding:12px 15px;
    border-radius:10px;
}

.component {
    background:#fff;
    border:1px solid var(--line);
    border-radius:18px;
    padding:18px;
    height:100%;
}

.component h3 {
    margin-top:0;
}

.badge {
    display:inline-block;
    border-radius:999px;
    background:#e0f2fe;
    color:#0369a1;
    padding:4px 10px;
    margin:2px;
    font-size:.8rem;
}

.small {
    color:var(--muted);
    font-size:.88rem;
}

@media(max-width:700px) {
    .hero { border-radius:18px; padding:21px 17px; }
    .card { padding:14px; }
    .block-container { padding-left:.55rem; padding-right:.55rem; }
    button { min-height:44px !important; }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# ابزارهای عمومی
# ============================================================

def fnum(x: float, digits: int = 5) -> str:
    if x is None:
        return "-"
    if not math.isfinite(float(x)):
        return "∞"
    return f"{x:.{digits}g}"


def eng(x: float, unit: str = "") -> str:
    if x == 0:
        return f"0 {unit}".strip()
    prefixes = [
        (1e9, "G"), (1e6, "M"), (1e3, "k"), (1, ""),
        (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"), (1e-12, "p")
    ]
    ax = abs(x)
    for factor, prefix in prefixes:
        if ax >= factor:
            return f"{x/factor:.5g} {prefix}{unit}".strip()
    return f"{x:.5g} {unit}".strip()


def formula(text: str):
    st.markdown(f'<div class="formula">{text}</div>', unsafe_allow_html=True)


def result(value: str, label: str):
    st.markdown(
        f'<div class="result"><div class="value">{value}</div>'
        f'<div class="label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def card(text: str):
    st.markdown(f'<div class="card">{text}</div>', unsafe_allow_html=True)


# ============================================================
# داده‌های قطعات
# ============================================================

TRANSISTORS = {
    "BC547": {
        "نوع": "NPN",
        "پایه‌ها": "C-B-E در پکیج رایج TO-92؛ حتماً دیتاشیت سازنده بررسی شود.",
        "مشخصات": "ترانزیستور سیگنال کوچک؛ VCE حدود 45V در نسخه‌های رایج.",
        "کاربرد": "تقویت سیگنال، سوئیچینگ و مدارهای آموزشی.",
    },
    "BC557": {
        "نوع": "PNP",
        "پایه‌ها": "ترتیب پایه‌ها به سازنده/پکیج وابسته است؛ دیتاشیت بررسی شود.",
        "مشخصات": "ترانزیستور PNP سیگنال کوچک؛ VCE حدود 45V در نسخه‌های رایج.",
        "کاربرد": "تقویت و سوئیچینگ سمت مثبت.",
    },
    "2N2222": {
        "نوع": "NPN",
        "پایه‌ها": "E-B-C در بسیاری از نسخه‌ها؛ بسته به سازنده متفاوت است.",
        "مشخصات": "ترانزیستور NPN عمومی برای سوئیچینگ و تقویت.",
        "کاربرد": "LED، رله، سوئیچینگ و تقویت.",
    },
    "2N3904": {
        "نوع": "NPN",
        "پایه‌ها": "E-B-C در نسخه‌های رایج؛ دیتاشیت قطعه ملاک است.",
        "مشخصات": "ترانزیستور NPN عمومی کم‌توان.",
        "کاربرد": "سوئیچینگ و تقویت سیگنال.",
    },
    "IRF540": {
        "نوع": "N-Channel MOSFET",
        "پایه‌ها": "G-D-S در پکیج رایج TO-220.",
        "مشخصات": "MOSFET قدرت با VDS نامی حدود 100V در نسخهٔ رایج.",
        "کاربرد": "سوئیچینگ بار و مدارهای قدرت.",
    },
}

ICS = {
    "LM358": {
        "نوع": "Dual Op-Amp",
        "پایه‌ها": "1OUT, 1IN-, 1IN+, V-, 2IN+, 2IN-, 2OUT, V+",
        "مشخصات": "دو تقویت‌کننده عملیاتی در یک آی‌سی 8 پایه.",
        "کاربرد": "تقویت، فیلتر، مقایسه و پردازش سیگنال.",
    },
    "LM741": {
        "نوع": "Op-Amp",
        "پایه‌ها": "Offset, IN-, IN+, V-, NC, OUT, V+, Offset",
        "مشخصات": "تقویت‌کننده عملیاتی کلاسیک؛ محدودیت‌های آن را باید در دیتاشیت دید.",
        "کاربرد": "آموزش مفاهیم Op-Amp.",
    },
    "LM7805": {
        "نوع": "Linear Regulator",
        "پایه‌ها": "IN-GND-OUT در پکیج رایج TO-220.",
        "مشخصات": "رگولاتور خطی 5V.",
        "کاربرد": "تولید 5V تنظیم‌شده از ورودی بالاتر.",
    },
    "NE555": {
        "نوع": "Timer",
        "پایه‌ها": "GND, TRIG, OUT, RESET, CTRL, THRESH, DISCH, VCC",
        "مشخصات": "تایمر بسیار رایج برای نوسان‌ساز و زمان‌سنج.",
        "کاربرد": "PWM، تایمر، مولد موج و چشمک‌زن.",
    },
    "CD4017": {
        "نوع": "Decade Counter",
        "پایه‌ها": "CLOCK, RESET, ENABLE و خروجی‌های Q0 تا Q9.",
        "مشخصات": "شمارنده ده‌دهی CMOS.",
        "کاربرد": "شمارنده، LED Chaser و تقسیم فرکانس.",
    },
}

DIODES = {
    "1N4007": {
        "نوع": "یکسوساز",
        "پایه‌ها": "Anode / Cathode",
        "مشخصات": "دیود یکسوساز عمومی.",
        "کاربرد": "یکسوسازی و حفاظت پلاریته.",
    },
    "1N4148": {
        "نوع": "سیگنال سریع",
        "پایه‌ها": "Anode / Cathode",
        "مشخصات": "دیود سیگنال سریع.",
        "کاربرد": "سوئیچینگ و مدارهای منطقی.",
    },
    "LED": {
        "نوع": "نورده",
        "پایه‌ها": "Anode (+) / Cathode (-)",
        "مشخصات": "Vf به رنگ و نوع LED وابسته است.",
        "کاربرد": "نمایش وضعیت و روشنایی.",
    },
    "Zener": {
        "نوع": "زنر",
        "پایه‌ها": "Anode / Cathode",
        "مشخصات": "در بایاس معکوس برای تثبیت/محدودسازی ولتاژ.",
        "کاربرد": "مرجع ولتاژ و حفاظت.",
    },
}

COLORS = {
    "مشکی": (0, 1, None, "#111827"),
    "قهوه‌ای": (1, 10, 1, "#78350f"),
    "قرمز": (2, 100, 2, "#dc2626"),
    "نارنجی": (3, 1000, None, "#f97316"),
    "زرد": (4, 10000, None, "#eab308"),
    "سبز": (5, 100000, .5, "#16a34a"),
    "آبی": (6, 1000000, .25, "#2563eb"),
    "بنفش": (7, 10000000, .1, "#7c3aed"),
    "خاکستری": (8, 100000000, .05, "#6b7280"),
    "سفید": (9, 1000000000, None, "#f8fafc"),
    "طلایی": (None, .1, 5, "#d4af37"),
    "نقره‌ای": (None, .01, 10, "#cbd5e1"),
}


# ============================================================
# تب 1 — ماشین‌حساب‌ها
# ============================================================

def tab_calculators():
    st.header("🧮 ماشین‌حساب‌های الکترونیک")

    tool = st.selectbox(
        "انتخاب ماشین‌حساب",
        ["قانون اهم", "کد رنگ مقاومت", "تقسیم ولتاژ",
         "مقاومت LED", "خازن", "فیلتر RC"],
        key="calc_tool",
    )
    st.divider()

    if tool == "قانون اهم":
        mode = st.radio(
            "چه مقداری محاسبه شود؟",
            ["ولتاژ V", "جریان I", "مقاومت R"],
            horizontal=True,
            key="ohm_mode",
        )

        a, b = st.columns(2)

        if mode == "ولتاژ V":
            with a:
                I = st.number_input("جریان (A)", min_value=0.0, value=.05, key="ohm_i")
            with b:
                R = st.number_input("مقاومت (Ω)", min_value=.001, value=100., key="ohm_r")
            V = I * R
            formula("V = I × R")
            result(eng(V, "V"), "ولتاژ")

        elif mode == "جریان I":
            with a:
                V = st.number_input("ولتاژ (V)", min_value=0.0, value=5., key="ohm_v2")
            with b:
                R = st.number_input("مقاومت (Ω)", min_value=.001, value=100., key="ohm_r2")
            I = V / R
            formula("I = V ÷ R")
            result(eng(I, "A"), "جریان")

        else:
            with a:
                V = st.number_input("ولتاژ (V)", min_value=0.0, value=5., key="ohm_v3")
            with b:
                I = st.number_input("جریان (A)", min_value=.000001, value=.05, key="ohm_i3")
            R = V / I
            formula("R = V ÷ I")
            result(eng(R, "Ω"), "مقاومت")

    elif tool == "کد رنگ مقاومت":
        bands = st.radio("نوع مقاومت", ["4 نوار", "5 نوار"], horizontal=True, key="band_type")
        n = 4 if bands == "4 نوار" else 5
        labels = ["نوار اول", "نوار دوم", "نوار سوم", "ضریب", "تلورانس"]
        if n == 4:
            labels = ["نوار اول", "نوار دوم", "ضریب", "تلورانس"]

        cols = st.columns(n)
        selected = []
        color_names = list(COLORS.keys())
        defaults4 = ["قهوه‌ای", "مشکی", "قهوه‌ای", "طلایی"]
        defaults5 = ["قهوه‌ای", "مشکی", "مشکی", "قهوه‌ای", "طلایی"]
        defaults = defaults4 if n == 4 else defaults5

        for i in range(n):
            with cols[i]:
                selected.append(
                    st.selectbox(
                        labels[i],
                        color_names,
                        index=color_names.index(defaults[i]),
                        key=f"band_{n}_{i}",
                    )
                )

        visual = '<div style="display:flex;justify-content:center;align-items:center;gap:7px;background:#f1f5f9;border-radius:16px;padding:20px;">'
        visual += '<div style="width:65px;height:26px;background:#d1a46c;border-radius:13px 0 0 13px;"></div>'
        for c in selected:
            visual += f'<div style="width:14px;height:46px;background:{COLORS[c][3]};border:1px solid #777;border-radius:3px;"></div>'
        visual += '<div style="width:65px;height:26px;background:#d1a46c;border-radius:0 13px 13px 0;"></div></div>'
        st.markdown(visual, unsafe_allow_html=True)

        try:
            if n == 4:
                d1, d2, mult, tol = [COLORS[x] for x in selected]
                value = (d1[0] * 10 + d2[0]) * mult[1]
                tolerance = tol[2]
            else:
                d1, d2, d3, mult, tol = [COLORS[x] for x in selected]
                value = (d1[0] * 100 + d2[0] * 10 + d3[0]) * mult[1]
                tolerance = tol[2]

            formula("R = ارقام اصلی × ضریب")
            result(
                f"{eng(value, 'Ω')}  ±{tolerance}%" if tolerance else eng(value, "Ω"),
                "مقدار مقاومت و تلورانس",
            )
        except Exception as e:
            st.error(f"خطا: {e}")

    elif tool == "تقسیم ولتاژ":
        a, b, c = st.columns(3)
        with a:
            Vin = st.number_input("Vin (V)", min_value=0., value=12., key="div_v")
        with b:
            R1 = st.number_input("R1 (Ω)", min_value=.001, value=10000., key="div_r1")
        with c:
            R2 = st.number_input("R2 (Ω)", min_value=.001, value=10000., key="div_r2")

        st.markdown(
            '<div class="circuit">Vin ── R1 ──●── R2 ── GND<br>             │<br>            Vout</div>',
            unsafe_allow_html=True,
        )
        Vout = Vin * R2 / (R1 + R2)
        formula("Vout = Vin × R2 / (R1 + R2)")
        result(eng(Vout, "V"), "ولتاژ خروجی")

    elif tool == "مقاومت LED":
        a, b, c = st.columns(3)
        with a:
            Vs = st.number_input("ولتاژ منبع (V)", min_value=0., value=5., key="led_vs")
        with b:
            Vf = st.number_input("ولتاژ LED (V)", min_value=0., value=2., key="led_vf")
        with c:
            ImA = st.number_input("جریان LED (mA)", min_value=.01, value=10., key="led_i")

        I = ImA / 1000
        R = (Vs - Vf) / I if I else 0

        st.markdown('<div class="circuit">+V ── R ── LED ── GND</div>', unsafe_allow_html=True)
        formula("R = (Vsource − VLED) ÷ ILED")

        if R <= 0:
            st.warning("ولتاژ منبع باید بیشتر از ولتاژ مستقیم LED باشد.")
        else:
            result(eng(R, "Ω"), "مقاومت لازم")

    elif tool == "خازن":
        operation = st.selectbox(
            "نوع محاسبه",
            ["سری", "موازی", "بار Q = CV", "انرژی E = ½CV²"],
            key="cap_op",
        )

        if operation in ["سری", "موازی"]:
            a, b = st.columns(2)
            with a:
                C1 = st.number_input("C1 (µF)", min_value=.000001, value=10., key="c1")
            with b:
                C2 = st.number_input("C2 (µF)", min_value=.000001, value=20., key="c2")

            if operation == "سری":
                C = C1 * C2 / (C1 + C2)
                formula("1/Ceq = 1/C1 + 1/C2")
            else:
                C = C1 + C2
                formula("Ceq = C1 + C2")

            result(eng(C * 1e-6, "F"), "ظرفیت معادل")

        elif operation == "بار Q = CV":
            a, b = st.columns(2)
            with a:
                C = st.number_input("C (µF)", min_value=0., value=100., key="cq")
            with b:
                V = st.number_input("V (V)", min_value=0., value=12., key="cv")
            Q = C * 1e-6 * V
            formula("Q = C × V")
            result(eng(Q, "C"), "بار خازن")

        else:
            a, b = st.columns(2)
            with a:
                C = st.number_input("C (µF)", min_value=0., value=100., key="ce")
            with b:
                V = st.number_input("V (V)", min_value=0., value=12., key="cev")
            E = .5 * C * 1e-6 * V**2
            formula("E = ½CV²")
            result(eng(E, "J"), "انرژی ذخیره‌شده")

    else:
        a, b = st.columns(2)
        with a:
            R = st.number_input("R (Ω)", min_value=.001, value=10000., key="rc_r")
        with b:
            C_uF = st.number_input("C (µF)", min_value=.000001, value=.1, key="rc_c")
        C = C_uF * 1e-6
        fc = 1 / (2 * math.pi * R * C)
        formula("fc = 1 ÷ (2πRC)")
        result(eng(fc, "Hz"), "فرکانس قطع")


# ============================================================
# تب 2 — مرجع قطعات
# ============================================================

def tab_components():
    st.header("🔧 مرجع قطعات")

    section = st.selectbox(
        "انتخاب بخش",
        ["ترانزیستورها", "آی‌سی‌ها", "دیودها", "مقاومت‌ها", "خازن‌ها", "جستجو"],
        key="component_section",
    )
    st.divider()

    if section == "ترانزیستورها":
        name = st.selectbox("قطعه", list(TRANSISTORS.keys()), key="tr")
        d = TRANSISTORS[name]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="component"><h3>{name}</h3>'
                f'<span class="badge">{d["نوع"]}</span>'
                f'<p><b>پایه‌ها:</b><br>{d["پایه‌ها"]}</p>'
                f'<p><b>مشخصات:</b><br>{d["مشخصات"]}</p>'
                f'<p><b>کاربرد:</b><br>{d["کاربرد"]}</p></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                '<div class="circuit">G / B ── پایه کنترل<br>'
                'D / C ── مسیر اصلی<br>S / E ── پایه مرجع</div>'
                '<div class="warn">⚠️ ترتیب دقیق پایه‌ها به پکیج و سازنده وابسته است؛ قبل از اتصال واقعی دیتاشیت را بررسی کنید.</div>',
                unsafe_allow_html=True,
            )

    elif section == "آی‌سی‌ها":
        name = st.selectbox("قطعه", list(ICS.keys()), key="ic")
        d = ICS[name]
        st.markdown(
            f'<div class="component"><h3>{name}</h3>'
            f'<span class="badge">{d["نوع"]}</span>'
            f'<p><b>Pinout:</b><br>{d["پایه‌ها"]}</p>'
            f'<p><b>مشخصات:</b><br>{d["مشخصات"]}</p>'
            f'<p><b>کاربرد:</b><br>{d["کاربرد"]}</p></div>',
            unsafe_allow_html=True,
        )

    elif section == "دیودها":
        name = st.selectbox("قطعه", list(DIODES.keys()), key="dio")
        d = DIODES[name]
        st.markdown(
            f'<div class="component"><h3>{name}</h3>'
            f'<span class="badge">{d["نوع"]}</span>'
            f'<p><b>پایه‌ها:</b> {d["پایه‌ها"]}</p>'
            f'<p><b>مشخصات:</b> {d["مشخصات"]}</p>'
            f'<p><b>کاربرد:</b> {d["کاربرد"]}</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="circuit">ANODE ── |>| ── CATHODE</div>', unsafe_allow_html=True)

    elif section == "مقاومت‌ها":
        series = st.radio("سری استاندارد", ["E12", "E24"], horizontal=True, key="eseries")
        e12 = [10,12,15,18,22,27,33,39,47,56,68,82]
        e24 = [10,11,12,13,15,16,18,20,22,24,27,30,33,36,39,43,47,51,56,62,68,75,82,91]
        values = e12 if series == "E12" else e24
        rows = []
        for x in values:
            rows.append({
                "مقدار پایه": x,
                "×1": f"{x} Ω",
                "×10": f"{x*10} Ω",
                "×100": f"{x*100} Ω",
                "×1k": f"{x} kΩ",
                "×10k": f"{x*10} kΩ",
                "×100k": f"{x*100} kΩ",
                "×1M": f"{x} MΩ",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    elif section == "خازن‌ها":
        st.markdown(
            """
            <div class="card">
            <h3>انواع خازن</h3>
            <p>🔹 <b>سرامیکی:</b> معمولاً بدون پلاریته و مناسب فرکانس‌های بالاتر.</p>
            <p>🔹 <b>الکترولیتی:</b> معمولاً پلاریته‌دار و مناسب ظرفیت‌های بالاتر.</p>
            <p>🔹 <b>تانتالیوم:</b> ظرفیت بالا در حجم کوچک؛ پلاریته و ولتاژ کاری مهم است.</p>
            <hr>
            <b>کدهای رایج سه‌رقمی:</b>
            101 = 100pF، 102 = 1nF، 103 = 10nF، 104 = 100nF، 224 = 220nF
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        q = st.text_input("جستجو", placeholder="مثلاً BC547 یا 555", key="search_component")
        cat = st.selectbox("فیلتر", ["همه", "ترانزیستور", "آی‌سی", "دیود"], key="search_cat")
        rows = []

        for name, d in TRANSISTORS.items():
            rows.append({"نام": name, "نوع": "ترانزیستور", "مشخصات": d["نوع"], "کاربرد": d["کاربرد"]})
        for name, d in ICS.items():
            rows.append({"نام": name, "نوع": "آی‌سی", "مشخصات": d["نوع"], "کاربرد": d["کاربرد"]})
        for name, d in DIODES.items():
            rows.append({"نام": name, "نوع": "دیود", "مشخصات": d["نوع"], "کاربرد": d["کاربرد"]})

        if cat != "همه":
            rows = [x for x in rows if x["نوع"] == cat]

        if q.strip():
            q = q.lower().strip()
            rows = [
                x for x in rows
                if q in x["نام"].lower()
                or q in x["مشخصات"].lower()
                or q in x["کاربرد"].lower()
            ]

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("قطعه‌ای پیدا نشد.")


# ============================================================
# تب 3 — آموزش تعاملی
# ============================================================

def tab_learning():
    st.header("🎓 آموزش تعاملی")

    lesson = st.selectbox(
        "انتخاب درس",
        ["سری و موازی", "AC و DC", "ترانزیستور به عنوان کلید",
         "تست جهت دیود", "شبیه‌ساز قانون اهم"],
        key="lesson",
    )
    st.divider()

    if lesson == "سری و موازی":
        mode = st.radio("مدار", ["سری", "موازی"], horizontal=True, key="sp")
        a, b, c = st.columns(3)
        with a:
            R1 = st.number_input("R1 (Ω)", min_value=.001, value=100., key="spr1")
        with b:
            R2 = st.number_input("R2 (Ω)", min_value=.001, value=200., key="spr2")
        with c:
            V = st.number_input("منبع (V)", min_value=0., value=10., key="spv")

        if mode == "سری":
            Req = R1 + R2
            I = V / Req
            V1 = I * R1
            V2 = I * R2
            st.markdown('<div class="circuit">+V ── R1 ── R2 ── GND</div>', unsafe_allow_html=True)
            formula("Req = R1 + R2")
            st.metric("مقاومت معادل", eng(Req, "Ω"))
            st.metric("جریان", eng(I, "A"))
            st.write(f"ولتاژ R1: **{eng(V1,'V')}**")
            st.write(f"ولتاژ R2: **{eng(V2,'V')}**")
            st.info("در مدار سری جریان در تمام اجزا یکسان است.")
        else:
            Req = R1 * R2 / (R1 + R2)
            I = V / Req
            I1 = V / R1
            I2 = V / R2
            st.markdown('<div class="circuit">+V ──┬── R1 ──┬── GND<br>        └── R2 ──┘</div>', unsafe_allow_html=True)
            formula("1/Req = 1/R1 + 1/R2")
            st.metric("مقاومت معادل", eng(Req, "Ω"))
            st.metric("جریان کل", eng(I, "A"))
            st.write(f"جریان R1: **{eng(I1,'A')}**")
            st.write(f"جریان R2: **{eng(I2,'A')}**")
            st.info("در مدار موازی ولتاژ دو سر شاخه‌ها یکسان است.")

    elif lesson == "AC و DC":
        mode = st.radio("نوع", ["DC", "AC"], horizontal=True, key="acdc")
        x = np.linspace(0, 2*np.pi, 300)
        if mode == "DC":
            y = np.ones_like(x) * 5
            st.line_chart(pd.DataFrame({"ولتاژ": y}, index=x))
            st.success("DC: مقدار و جهت سیگنال در حالت ایده‌آل ثابت است.")
            st.write("مثال: باتری، پاوربانک و خروجی USB.")
        else:
            y = 5*np.sin(x)
            st.line_chart(pd.DataFrame({"ولتاژ": y}, index=x))
            st.info("AC: مقدار و/یا جهت سیگنال با زمان تغییر می‌کند.")
            st.write("مثال: برق شهر و سیگنال‌های متناوب.")

    elif lesson == "ترانزیستور به عنوان کلید":
        a, b, c = st.columns(3)
        with a:
            Vin = st.number_input("ولتاژ کنترل (V)", min_value=0., value=5., key="swvin")
        with b:
            RB = st.number_input("مقاومت پایه (Ω)", min_value=1., value=10000., key="swrb")
        with c:
            beta = st.number_input("β تقریبی", min_value=1., value=100., key="swbeta")
        VBE = .7
        IB = max(0., (Vin - VBE) / RB)
        IC = beta * IB
        st.markdown('<div class="circuit">کنترل ── RB ── B<br>                 C ── بار<br>                 E ── GND</div>', unsafe_allow_html=True)
        formula("IB ≈ (Vin − VBE) / RB")
        formula("IC ≈ β × IB")
        st.metric("جریان پایه", eng(IB, "A"))
        st.metric("جریان کلکتور تقریبی", eng(IC, "A"))
        st.warning("این محاسبه آموزشی است؛ برای سوئیچ واقعی باید اشباع و محدودیت توان بررسی شود.")

    elif lesson == "تست جهت دیود":
        direction = st.radio("اتصال", ["بایاس مستقیم", "بایاس معکوس"], horizontal=True, key="diode_dir")
        st.markdown('<div class="circuit">+ ─── |>| ─── −</div>' if direction == "بایاس مستقیم"
                    else '<div class="circuit">+ ─── |<| ─── −</div>', unsafe_allow_html=True)
        if direction == "بایاس مستقیم":
            st.success("در بایاس مستقیم، دیود در شرایط مناسب جریان عبور می‌دهد.")
        else:
            st.info("در بایاس معکوس، دیود معمولی تا قبل از شکست جریان بسیار کمی عبور می‌دهد.")

    else:
        V = st.slider("ولتاژ V", 0.0, 24.0, 5.0, .1, key="simv")
        R = st.slider("مقاومت R (Ω)", 1.0, 10000.0, 100.0, 1.0, key="simr")
        I = V / R
        formula("I = V ÷ R")
        result(eng(I, "A"), "جریان لحظه‌ای")
        st.progress(min(V / 24, 1.0))
        st.write(f"ولتاژ: **{V:.2f} V**")
        st.write(f"مقاومت: **{eng(R,'Ω')}**")


# ============================================================
# RAG — Gemini + FAISS + PDF
# ============================================================

def gemini_key() -> Optional[str]:
    try:
        value = st.secrets.get("GEMINI_API_KEY")
        if value:
            return value
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")


@st.cache_resource(show_spinner=False)
def get_client():
    key = gemini_key()
    if not GEMINI_OK or not key:
        return None
    return genai.Client(api_key=key)


def normalize(text: str) -> str:
    for a, b in {"ي":"ی", "ى":"ی", "ك":"ک", "\u200c":" "}.items():
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip()


def chunks(text: str, size: int = 1000, overlap: int = 150) -> List[str]:
    text = normalize(text)
    out = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = end - overlap
    return out


def extract_pdf(file) -> List[Dict[str, Any]]:
    if not PYPDF_OK:
        raise RuntimeError("pypdf نصب نشده است.")
    reader = PdfReader(io.BytesIO(file.getvalue()))
    out = []
    for page_no, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        for i, piece in enumerate(chunks(text)):
            out.append({
                "text": piece,
                "page": page_no,
                "source": file.name,
                "chunk": i + 1,
            })
    return out


def embed(client, texts: List[str], task: str) -> np.ndarray:
    vectors = []
    for start in range(0, len(texts), 16):
        batch = texts[start:start+16]
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task,
                output_dimensionality=768,
            ),
        )
        vectors.extend([x.values for x in response.embeddings])
    return np.asarray(vectors, dtype=np.float32)


def build_index(file):
    client = get_client()
    if client is None:
        raise RuntimeError("Gemini API Key تنظیم نشده است.")
    records = extract_pdf(file)
    if not records:
        raise RuntimeError("از PDF متن قابل استخراج پیدا نشد.")
    vectors = embed(client, [x["text"] for x in records], "RETRIEVAL_DOCUMENT")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    st.session_state["rag_records"] = records
    st.session_state["rag_index"] = index
    st.session_state["rag_name"] = file.name
    return len(records)


def rag_search(question: str, k: int = 5):
    client = get_client()
    if client is None or "rag_index" not in st.session_state:
        return []
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    qv = np.asarray([response.embeddings[0].values], dtype=np.float32)
    faiss.normalize_L2(qv)
    scores, ids = st.session_state["rag_index"].search(qv, k)
    out = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            item = dict(st.session_state["rag_records"][idx])
            item["score"] = float(score)
            out.append(item)
    return out


def answer_from_context(question: str, results: List[Dict[str, Any]]):
    client = get_client()
    context = "\n\n---\n\n".join(
        f"منبع: {x['source']} | صفحه: {x['page']}\n{x['text']}"
        for x in results
    )
    prompt = f"""
تو دستیار آموزشی الکترونیک برای هنرجوی پایه دهم هنرستان هستی.

سؤال:
{question}

منابع پیدا شده:
{context}

قوانین:
- فارسی و ساده پاسخ بده.
- فقط وقتی ادعایی را به کتاب نسبت می‌دهی از متن بالا استفاده کن.
- شماره صفحه را دقیقاً از metadata بالا ذکر کن.
- اگر منبع کافی نیست، صریحاً بگو.
- فرمول‌ها را واضح بنویس.
- چیزی را جعل نکن.
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=.2,
            max_output_tokens=1200,
        ),
    )
    return response.text


def general_answer(question: str):
    client = get_client()
    if client is None:
        return None
    prompt = f"""
تو معلم الکترونیک هنرستان هستی.
به سؤال زیر فارسی، دقیق و مناسب هنرجوی پایه دهم پاسخ بده.
اگر فرمول لازم است بنویس.
اگر موضوع ایمنی برق است هشدار مناسب بده.
سؤال:
{question}
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=.25,
            max_output_tokens=1000,
        ),
    )
    return response.text


def tab_ai():
    st.header("🤖 دستیار هوشمند")

    if not GEMINI_OK:
        st.error("کتابخانه google-genai نصب نشده است.")
        return
    if not FAISS_OK:
        st.error("کتابخانه FAISS نصب نشده است.")
        return
    if not PYPDF_OK:
        st.error("کتابخانه pypdf نصب نشده است.")
        return
    if not gemini_key():
        st.warning("برای فعال شدن AI، GEMINI_API_KEY را در Streamlit Secrets قرار دهید.")
        st.markdown(
            '<div class="note">کلید API را داخل کد یا HTML عمومی قرار ندهید؛ '
            'در غیر این صورت هر بازدیدکننده می‌تواند آن را مشاهده و مصرف کند.</div>',
            unsafe_allow_html=True,
        )
        return

    uploaded = st.file_uploader(
        "PDF کتاب الکترونیک را بارگذاری کنید",
        type=["pdf"],
        key="pdf_upload",
    )

    if uploaded:
        if st.button("📚 پردازش کتاب", type="primary", key="process"):
            try:
                with st.spinner("در حال استخراج متن، chunk بندی و ساخت embedding..."):
                    n = build_index(uploaded)
                st.success(f"کتاب آماده شد؛ {n} قطعه متنی ساخته شد.")
            except Exception as e:
                st.error(f"خطا در پردازش PDF: {e}")

    if "rag_index" in st.session_state:
        st.success(
            f"منبع فعال: {st.session_state.get('rag_name','-')} | "
            f"{len(st.session_state['rag_records'])} قطعه"
        )

    question = st.text_area(
        "سؤال خود را بنویسید",
        placeholder="مثلاً: قانون تقسیم ولتاژ چگونه کار می‌کند؟",
        height=110,
        key="ai_q",
    )

    if st.button("💬 پاسخ بده", type="primary", key="ask"):
        if not question.strip():
            st.warning("سؤال را وارد کنید.")
            return

        try:
            with st.spinner("در حال جستجو در کتاب..."):
                hits = rag_search(question, 5)

            # آستانه نسبتاً محافظه‌کارانه برای جلوگیری از نسبت دادن متن نامرتبط به کتاب
            if hits and hits[0]["score"] >= .35:
                with st.spinner("در حال تولید پاسخ بر اساس کتاب..."):
                    ans = answer_from_context(question, hits)
                st.markdown("### پاسخ")
                st.markdown(ans)
                st.markdown("### 📖 منابع")
                for h in hits[:3]:
                    st.caption(
                        f"📄 {h['source']} — صفحه {h['page']} — "
                        f"امتیاز شباهت: {h['score']:.2f}"
                    )
            else:
                st.info("منبع مناسبی در PDF پیدا نشد؛ از دانش عمومی Gemini استفاده می‌شود.")
                with st.spinner("در حال پاسخ‌گویی..."):
                    ans = general_answer(question)
                if ans:
                    st.markdown("### پاسخ")
                    st.markdown(ans)
                else:
                    st.error("پاسخ دریافت نشد.")
        except Exception as e:
            st.error(f"خطا در دستیار هوشمند: {e}")


# ============================================================
# تب 5 — ابزارهای اضافی
# ============================================================

def tab_tools():
    st.header("🛠️ ابزارهای اضافی")

    tool = st.selectbox(
        "ابزار",
        ["تبدیل واحد", "جدول کد رنگی مقاومت", "مبدل کد خازن"],
        key="extra",
    )
    st.divider()

    if tool == "تبدیل واحد":
        factors = {
            "مقاومت": {"Ω":1, "kΩ":1e3, "MΩ":1e6},
            "خازن": {"pF":1e-12, "nF":1e-9, "µF":1e-6, "mF":1e-3, "F":1},
            "ولتاژ": {"mV":1e-3, "V":1, "kV":1e3},
            "جریان": {"µA":1e-6, "mA":1e-3, "A":1},
        }
        kind = st.selectbox("کمیت", list(factors), key="ukind")
        units = list(factors[kind])
        a,b,c = st.columns(3)
        with a:
            value = st.number_input("مقدار", value=1., key="uval")
        with b:
            source = st.selectbox("از", units, key="usource")
        with c:
            target = st.selectbox("به", units, key="utarget", index=min(1,len(units)-1))
        base = value * factors[kind][source]
        out = base / factors[kind][target]
        result(f"{fnum(out)} {target}", f"{value:g} {source}")

    elif tool == "جدول کد رنگی مقاومت":
        rows = []
        for name, (digit, mult, tol, _) in COLORS.items():
            rows.append({
                "رنگ": name,
                "رقم": "-" if digit is None else digit,
                "ضریب": mult,
                "تلورانس": "-" if tol is None else f"±{tol}%",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ دانلود جدول CSV",
            df.to_csv(index=False).encode("utf-8-sig"),
            "جدول_کد_رنگ_مقاومت.csv",
            "text/csv",
            key="download_colors",
        )

    else:
        code = st.text_input("کد سه‌رقمی خازن", value="104", max_chars=3, key="ccode")
        if re.fullmatch(r"\d{3}", code):
            a, b, m = map(int, code)
            pf = (a*10+b) * 10**m
            formula("دو رقم اول × 10^(رقم سوم) = pF")
            result(eng(pf*1e-12, "F"), f"{code} ≈ {pf:g} pF")
        else:
            st.warning("کد باید دقیقاً سه رقم باشد.")


# ============================================================
# صفحه اصلی
# ============================================================

st.markdown("""
<div class="hero">
    <h1>⚡ الکترونیک‌یار</h1>
    <p>
        ابزار آموزشی رایگان و فارسی برای هنرجویان الکترونیک هنرستان
        <br>
        محاسبات • مرجع قطعات • آموزش تعاملی • دستیار هوشمند
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ وضعیت سیستم"):
    a,b,c,d = st.columns(4)
    a.metric("ماشین‌حساب‌ها", "فعال")
    b.metric("مرجع قطعات", "فعال")
    c.metric("FAISS", "فعال" if FAISS_OK else "غیرفعال")
    d.metric("Gemini", "فعال" if gemini_key() else "تنظیم نشده")

tabs = st.tabs([
    "🧮 ماشین‌حساب‌ها",
    "🔧 مرجع قطعات",
    "🎓 آموزش تعاملی",
    "🤖 دستیار هوشمند",
    "🛠️ ابزارهای اضافی",
])

with tabs[0]:
    tab_calculators()
with tabs[1]:
    tab_components()
with tabs[2]:
    tab_learning()
with tabs[3]:
    tab_ai()
with tabs[4]:
    tab_tools()

st.divider()
st.markdown(
    '<div style="text-align:center;color:#64748b;padding:18px">'
    '<b>الکترونیک‌یار</b><br>'
    'ابزار آموزشی رایگان برای یادگیری الکترونیک<br><br>'
    '⚠️ برای مدار واقعی، ولتاژ، جریان، توان و محدودیت قطعات را بررسی کنید.'
    '</div>',
    unsafe_allow_html=True,
)
