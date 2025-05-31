import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Metode Numerik", layout="centered")
st.title("Perhitungan Metode Numerik")

method = st.selectbox("Pilih metode:", ["Bagi Dua", "Regula Falsi", "Iterasi Titik Tetap", "Newton-Raphson", "Secant"])

f_str = st.text_input("Masukkan fungsi f(x):", "x**2 - 4")
x = sp.symbols('x')
f = sp.sympify(f_str)
f_lambd = sp.lambdify(x, f, modules=['numpy'])

x0 = st.number_input("Nilai awal x0 / a:", value=1.0)
x1 = st.number_input("Nilai awal x1 / b:", value=3.0)
tol = st.number_input("Toleransi error:", value=0.0001, format="%f")
max_iter = st.number_input("Maksimum iterasi:", value=50, step=1)

if st.button("Hitung"):
    hasil = []
    if method == "Bagi Dua":
        a, b = x0, x1
        for i in range(int(max_iter)):
            c = (a + b) / 2
            fc = f_lambd(c)
            hasil.append((i+1, c, fc))
            if abs(fc) < tol:
                break
            if f_lambd(a) * fc < 0:
                b = c
            else:
                a = c

    elif method == "Regula Falsi":
        a, b = x0, x1
        for i in range(int(max_iter)):
            fa, fb = f_lambd(a), f_lambd(b)
            x2 = b - fb * (b - a) / (fb - fa)
            fx2 = f_lambd(x2)
            hasil.append((i+1, x2, fx2))
            if abs(fx2) < tol:
                break
            if fa * fx2 < 0:
                b = x2
            else:
                a = x2

    elif method == "Iterasi Titik Tetap":
        g_str = st.text_input("Masukkan fungsi g(x) untuk titik tetap:", "np.sqrt(4)")
        g = eval("lambda x: " + g_str)
        x_curr = x0
        for i in range(int(max_iter)):
            x_next = g(x_curr)
            err = abs(x_next - x_curr)
            hasil.append((i+1, x_next, err))
            if err < tol:
                break
            x_curr = x_next

    elif method == "Newton-Raphson":
        df = sp.diff(f, x)
        df_lambd = sp.lambdify(x, df, modules=['numpy'])
        x_curr = x0
        for i in range(int(max_iter)):
            dfx = df_lambd(x_curr)
            if dfx == 0:
                st.error("Turunan nol, metode gagal.")
                break
            x_next = x_curr - f_lambd(x_curr)/dfx
            err = abs(x_next - x_curr)
            hasil.append((i+1, x_next, err))
            if err < tol:
                break
            x_curr = x_next

    elif method == "Secant":
        x_prev, x_curr = x0, x1
        for i in range(int(max_iter)):
            f_prev, f_curr = f_lambd(x_prev), f_lambd(x_curr)
            if f_curr - f_prev == 0:
                st.error("Pembagi nol, metode gagal.")
                break
            x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            err = abs(x_next - x_curr)
            hasil.append((i+1, x_next, err))
            if err < tol:
                break
            x_prev, x_curr = x_curr, x_next

    st.subheader("Hasil Iterasi")
    if method in ["Bagi Dua", "Regula Falsi"]:
        df = pd.DataFrame(hasil, columns=["Iterasi", "x", "f(x)"])
    else:
        df = pd.DataFrame(hasil, columns=["Iterasi", "x", "Error"])
    st.dataframe(df.style.format({"x": "{:.6f}", df.columns[-1]: "{:.6f}"}))
