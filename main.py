import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Metode Numerik", layout="centered")
st.title("Perhitungan Metode Numerik")

# Pilih metode
method = st.selectbox("Pilih metode:", ["Bagi Dua", "Regula Falsi", "Iterasi Titik Tetap", "Newton-Raphson", "Secant"])

# Input fungsi dari user
f_str = st.text_input("Masukkan fungsi f(x):", "x**2 - 4")

# Fungsi aman menggunakan eval
def f_lambd(x):
    try:
        return eval(f_str, {"x": x, "np": np, "__builtins__": {}})
    except:
        return np.nan

# Parameter umum
x0 = st.number_input("Nilai awal x0 / a:", value=1.0)
x1 = st.number_input("Nilai awal x1 / b:", value=3.0)
tol = st.number_input("Toleransi error:", value=0.0001, format="%f")
max_iter = st.number_input("Maksimum iterasi:", value=50, step=1)

# Jalankan metode
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
        g_str = st.text_input("Masukkan fungsi g(x) untuk titik tetap:", "(x + 4/x)/2")
        g = lambda x: eval(g_str, {"x": x, "np": np, "__builtins__": {}})
        x_curr = x0
        for i in range(int(max_iter)):
            try:
                x_next = g(x_curr)
                err = abs(x_next - x_curr)
                hasil.append((i+1, x_next, err))
                if err < tol:
                    break
                x_curr = x_next
            except:
                st.error("Error saat evaluasi fungsi g(x)")
                break

    elif method == "Newton-Raphson":
        dfdx_str = st.text_input("Masukkan turunan f(x):", "2*x")
        f_prime = lambda x: eval(dfdx_str, {"x": x, "np": np, "__builtins__": {}})
        x_curr = x0
        for i in range(int(max_iter)):
            fx = f_lambd(x_curr)
            dfx = f_prime(x_curr)
            if dfx == 0:
                st.error("Turunan nol, metode gagal.")
                break
            x_next = x_curr - fx/dfx
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

    # Tampilkan hasil dalam bentuk tabel
    st.subheader("Hasil Iterasi")
    if method in ["Bagi Dua", "Regula Falsi"]:
        df = pd.DataFrame(hasil, columns=["Iterasi", "x", "f(x)"])
    else:
        df = pd.DataFrame(hasil, columns=["Iterasi", "x", "Error"])
    st.dataframe(df.style.format({"x": "{:.6f}", df.columns[-1]: "{:.6f}"}))
