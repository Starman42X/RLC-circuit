import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tkinter as tk
from tkinter import simpledialog

def get_user_input():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    def ask(title, prompt, initial, cast=float):
        try:
            val = simpledialog.askstring(title, prompt, initialvalue=str(initial))
            if val is None: return initial
            return cast(val)
        except ValueError:
            return initial

    circuit_type = ask("Circuit Type", "Circuit Type ('series' or 'parallel'):", "series", str).lower()
    if circuit_type not in ['series', 'parallel']:
        circuit_type = 'series'

    R = ask("Parameters", "Resistance R (Ohm):", 100.0)
    L_mH = ask("Parameters", "Inductance L (mH):", 10.0)
    C_uF = ask("Parameters", "Capacitance C (µF):", 1.0)
    U_gen = ask("Parameters", "Generator Voltage U_gen (V):", 1.0)
    f_min = ask("Frequency", "Start Frequency f_min (Hz):", 10.0)
    f_max = ask("Frequency", "End Frequency f_max (Hz):", 50000.0)
    num_points = ask("Frequency", "Number of points:", 200, int)
    Ri = ask("Parameters", "Internal Resistance Ri (Ohm) [default 0]:", 0.0)

    root.destroy()
    return circuit_type, R, L_mH * 1e-3, C_uF * 1e-6, U_gen, f_min, f_max, num_points, Ri

def generate_theoretical_data(ctype, R, L, C, U_gen, f_min, f_max, num_points, Ri):
    # Resonance frequency
    f0 = 1 / (2 * np.pi * np.sqrt(L * C))

    # Create frequency array with logspace + linspace around resonance
    f_log = np.logspace(np.log10(max(f_min, 1e-1)), np.log10(f_max), num_points // 2)
    
    # Linear spacing around f0
    span = f0 * 0.5
    f_lin = np.linspace(max(f_min, f0 - span), min(f_max, f0 + span), num_points // 2)
    
    f = np.sort(np.unique(np.concatenate((f_log, f_lin))))
    w = 2 * np.pi * f

    if ctype == 'series':
        # Series RLC
        # Z = R + j(wL - 1/wC)
        # I = U / (Ri + Z)
        
        X = w * L - 1 / (w * C)
        Z = R + 1j * X  # Impedance of the RLC circuit itself
        Z_total = (R + Ri) + 1j * X
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            I = U_gen / Z_total
            
        Delta_f = (R + Ri) / (2 * np.pi * L)

    else:
        # Parallel RLC
        # Z_total = 1 / (1/R + 1/(jwL) + jwC)
        # I = U_gen / Z_total (assuming Ri is negligible/handled or part of source current limit if Ri added)
        
        # Admittance
        with np.errstate(divide='ignore', invalid='ignore'):
            Y = 1/R + 1/(1j * w * L) + 1j * w * C
            Z = 1 / Y
        
        # If Ri is provided, it typically adds to series impedance for current calculation
        if Ri > 0:
            Z_total = Ri + Z
            with np.errstate(divide='ignore', invalid='ignore'):
                I = U_gen / Z_total
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                I = U_gen / Z
        
        Delta_f = 1 / (2 * np.pi * R * C)

    return f, Z, I, f0, Delta_f

def plot_locus_curves(ctype, f, Z, I, R, L, C, f0, Delta_f, U_gen, Ri):
    # Figure settings
    fig_width = 19.2
    fig_height = 10.8
    
    suffix = f"_{ctype}_RLC"
    title_suffix = f" ({ctype.capitalize()} RLC: R={R}Ω, L={L*1e3:.1f}mH, C={C*1e6:.1f}µF)"

    # --- Plot Z Locus ---
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    Z_real = np.real(Z)
    Z_imag = np.imag(Z)
    
    colors = cm.rainbow(np.linspace(0, 1, len(f)))
    
    for i in range(len(f)):
        label = None
        if i == 0:
            label = r'$Z(\omega_{start})$' + f' ({f[i]:.1f} Hz)'
        elif i == len(f) - 1:
            label = r'$Z(\omega_{end})$' + f' ({f[i]:.1f} Hz)'
        
        if label:
            ax.plot([0, Z_real[i]], [0, Z_imag[i]], color=colors[i], marker='o', markersize=5, label=label)
        else:
            # Thin lines for intermediate points to avoid clutter but matching style
            ax.plot([0, Z_real[i]], [0, Z_imag[i]], color=colors[i], marker='o', markersize=3, alpha=0.4)

    ax.set_xlabel(r'Reelle Achse [$\Omega$]')
    ax.set_ylabel(r'Imaginäre Achse [$\Omega$]')
    ax.set_title(fr'Ortskurve des komplexen Widerstands $Z(\omega)${title_suffix}')
    ax.grid(True)
    ax.legend(loc='best')
    
    filename_z = f"{ctype}_RLC_Z_locus.png"
    plt.savefig(filename_z)
    plt.close()
    print(f"Saved {filename_z}")

    # --- Plot I Locus ---
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    I_ma = I * 1000
    I_real = np.real(I_ma)
    I_imag = np.imag(I_ma)
    
    for i in range(len(f)):
        label = None
        if i == 0:
            label = r'$I(\omega_{start})$'
        elif i == len(f) - 1:
            label = r'$I(\omega_{end})$'
        
        if label:
            ax.plot([0, I_real[i]], [0, I_imag[i]], color=colors[i], marker='o', markersize=5, label=label)
        else:
            ax.plot([0, I_real[i]], [0, I_imag[i]], color=colors[i], marker='o', markersize=3, alpha=0.4)

    # Dashed connection line
    ax.plot(I_real, I_imag, 'k--', label='Ortskurve (verbunden)')
    
    ax.set_xlabel(r'Re($I$) [mA]')
    ax.set_ylabel(r'Im($I$) [mA]')
    ax.set_title(fr'Ortskurve des komplexen Stromes $I(\omega)${title_suffix}')
    ax.grid(True)
    ax.legend(loc='best')
    
    filename_i = f"{ctype}_RLC_I_locus.png"
    plt.savefig(filename_i)
    plt.close()
    print(f"Saved {filename_i}")

if __name__ == "__main__":
    # Get inputs
    ctype, R, L, C, U, f_min, f_max, n, Ri = get_user_input()
    
    # Generate data
    f, Z, I, f0, Delta_f = generate_theoretical_data(ctype, R, L, C, U, f_min, f_max, n, Ri)
    
    # Plot
    plot_locus_curves(ctype, f, Z, I, R, L, C, f0, Delta_f, U, Ri)
    
    print("Done. Plots generated.")
