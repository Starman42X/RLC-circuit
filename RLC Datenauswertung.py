import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from tkinter import filedialog, simpledialog, Tk

def compute_params(filename):
    """
    Computes R, L, C with Gaussian error propagation from CSV data for series RLC circuit.

    Assumptions:
    - Circuit is series RLC.
    - U = 1V
    - Data may not be sorted; sort by frequency.
    - Use closest measured points to I_max / sqrt(2) for bandwidth (no interpolation).
    - Uncertainties: Delta I dynamically from std near max, Delta f = half the average spacing to neighbors.

    Returns: R, Delta_R, L, Delta_L, C, Delta_C, f0, Delta_f, df_sorted
    """
    df = pd.read_csv(filename, encoding='latin1')

    # Sort by frequency
    df = df.sort_values(by='f / Hz').reset_index(drop=True)

    f = df['f / Hz'].values
    I_ma = df['I / mA'].values
    # Flexible column search for phi (to handle variations like 'phi / °' or similar)
    phi_col = [col for col in df.columns if 'phi' in col.lower()][0]
    phi_deg = df[phi_col].values
    UR = df['UR / V'].values
    ULC = df['ULC / V'].values  # Assuming UX = ULC

    I_amp = I_ma / 1000.0

    # Find resonance: max I
    idx_res = np.argmax(I_amp)
    f0 = f[idx_res]
    I_max = I_amp[idx_res]

    # Estimate Delta f0: half the span to neighbors
    if idx_res > 0 and idx_res < len(f) - 1:
        delta_f0 = 0.5 * min(f[idx_res] - f[idx_res-1], f[idx_res+1] - f[idx_res])
    elif idx_res > 0:
        delta_f0 = 0.5 * (f[idx_res] - f[idx_res-1])
    elif idx_res < len(f) - 1:
        delta_f0 = 0.5 * (f[idx_res+1] - f[idx_res])
    else:
        delta_f0 = 1.0  # Minimal uncertainty if single point

    # Dynamic window_size: Based on number of points near resonance (e.g., where I > 0.9 * I_max)
    near_res_mask = I_amp > 0.9 * I_max
    num_near_points = np.sum(near_res_mask)
    window_size = max(1, num_near_points // 2)  # At least 1, scaled to data density

    # Dynamic delta_I_max: std of I near resonance (using dynamic window)
    idx_start = max(0, idx_res - window_size)
    idx_end = min(len(I_amp), idx_res + window_size + 1)
    I_window = I_amp[idx_start:idx_end]
    delta_I_max = np.std(I_window) if len(I_window) > 1 else 1e-6  # Fallback if single point

    # R = 1 / I_max (in ohms, since U=1V)
    R = 1.0 / I_max
    delta_R = (1.0 / I_max**2) * delta_I_max  # Gaussian: |dR/dI_max| * delta_I_max

    # Target I for half-power: I_max / sqrt(2)
    I_target = I_max / np.sqrt(2)

    # Find closest points below and above f0 for I closest to I_target
    lower_mask = f < f0
    if not np.any(lower_mask):
        raise ValueError(f"Keine Daten unterhalb der Resonanz in {filename}")
    f_lower = f[lower_mask]
    I_lower = I_amp[lower_mask]
    idx_lower = np.argmin(np.abs(I_lower - I_target))
    f1 = f_lower[idx_lower]
    I1 = I_lower[idx_lower]

    upper_mask = f > f0
    if not np.any(upper_mask):
        raise ValueError(f"Keine Daten oberhalb der Resonanz in {filename}")
    f_upper = f[upper_mask]
    I_upper = I_amp[upper_mask]
    idx_upper = np.argmin(np.abs(I_upper - I_target))
    f2 = f_upper[idx_upper]
    I2 = I_upper[idx_upper]

    Delta_f = f2 - f1

    # Estimate Delta f1 and Delta f2: half spacing to neighbors
    idx_l_global = np.where(f == f1)[0][0]
    if idx_l_global > 0 and idx_l_global < len(f) - 1:
        delta_f1 = 0.5 * min(f[idx_l_global] - f[idx_l_global-1], f[idx_l_global+1] - f[idx_l_global])
    else:
        delta_f1 = 1.0

    idx_u_global = np.where(f == f2)[0][0]
    if idx_u_global > 0 and idx_u_global < len(f) - 1:
        delta_f2 = 0.5 * min(f[idx_u_global] - f[idx_u_global-1], f[idx_u_global+1] - f[idx_u_global])
    else:
        delta_f2 = 1.0

    delta_Delta_f = np.sqrt(delta_f1**2 + delta_f2**2)  # Gaussian for Delta_f = f2 - f1

    # L = R / (2 pi Delta_f) in H
    pi = np.pi
    k = 2 * pi * Delta_f
    L = R / k
    # Gaussian error propagation for L = R / k: delta_L / L = sqrt( (delta_R / R)^2 + (delta_k / k)^2 )
    # Since k = 2*pi*Delta_f, delta_k / k = delta_Delta_f / Delta_f (2*pi constant, no error)
    delta_L = L * np.sqrt((delta_R / R)**2 + (delta_Delta_f / Delta_f)**2)

    # C = 1 / (4 pi^2 f0^2 L) in F
    m = 4 * pi**2 * f0**2 * L
    C = 1 / m
    # Gaussian for C = 1 / m: delta_C / C = delta_m / m
    # m = const * f0^2 * L, delta_m / m = sqrt( (2 * delta_f0 / f0)^2 + (delta_L / L)^2 )
    delta_C = C * np.sqrt((delta_L / L)**2 + (2 * delta_f0 / f0)**2)

    return R, delta_R, L, delta_L, C, delta_C, f0, Delta_f, df

def generate_theoretical_df(f_min, f_max, num_points, R_theo, L_theo, C_theo, U=1.0):
    """
    Generate a DataFrame with theoretical values for the given parameters.
    """
    # Use more points for smoother plot
    f_log = np.logspace(np.log10(f_min), np.log10(f_max), num_points // 2)
    # Add linear spaced points around resonance for balance
    f0_theo = 1 / (2 * np.pi * np.sqrt(L_theo * C_theo))
    f_lin = np.linspace(max(f_min, f0_theo / 10), min(f_max, f0_theo * 10), num_points // 2)
    f = np.sort(np.unique(np.concatenate((f_log, f_lin))))

    omega = 2 * np.pi * f
    X = omega * L_theo - 1 / (omega * C_theo)
    Z_mag = np.sqrt(R_theo**2 + X**2)
    phi_rad = np.arctan2(X, R_theo)
    phi_deg = phi_rad * 180 / np.pi

    I_amp = U / Z_mag
    I_ma = I_amp * 1000

    UR = I_amp * R_theo
    ULC = I_amp * np.abs(X)  # Magnitude of reactive voltage

    df_theo = pd.DataFrame({
        'f / Hz': f,
        'I / mA': I_ma,
        'phi / °': phi_deg,
        'UR / V': UR,
        'ULC / V': ULC
    })
    return df_theo

def plot_data(df, filename, f0, Delta_f, R, L, C, theoretical=False):
    suffix = "_theoretisch" if theoretical else ""
    f = df['f / Hz'].values
    I_ma = df['I / mA'].values
    # Flexible column search for phi
    phi_col = [col for col in df.columns if 'phi' in col.lower()][0]
    phi_deg = df[phi_col].values
    UR = df['UR / V'].values
    ULC = df['ULC / V'].values  # UX = ULC

    I_amp = I_ma / 1000.0
    Z_mag = 1.0 / I_amp
    phi_rad = phi_deg * np.pi / 180.0
    Z_real = Z_mag * np.cos(phi_rad)
    Z_imag = Z_mag * np.sin(phi_rad)

    I_real = I_ma * np.cos(-phi_rad)
    I_imag = I_ma * np.sin(-phi_rad)

    # Set figure size for 1920x1080 (in inches, assuming 100 dpi)
    fig_width = 19.2
    fig_height = 10.8

    # Calculate fo and fu
    fu = f0 - Delta_f / 2
    fo = f0 + Delta_f / 2

    # Plot I(f) only with annotations
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlabel(r'Frequenz $f$ [Hz]')
    ax.set_ylabel(r'Strom $I$ [mA]')
    ax.plot(f, I_ma, 'b-', label=r'$I(f)$')
    # Mark f0
    ax.axvline(x=f0, color='r', linestyle='--', label=r'$f_0 = {:.2f}$ Hz'.format(f0))
    # Mark fo and fu with different colors
    ax.axvline(x=fo, color='g', linestyle='--', label=r'$f_o = {:.2f}$ Hz'.format(fo))
    ax.axvline(x=fu, color='purple', linestyle='--', label=r'$f_u = {:.2f}$ Hz'.format(fu))
    # Shade the bandwidth area
    ax.axvspan(fu, fo, alpha=0.2, color='grey', label=r'Bandbreite $b = {:.2f}$ Hz'.format(Delta_f))
    ax.grid(True)
    fig.suptitle(fr'Frequenzgang des Stromes $I(f)$ für {filename} {suffix}')
    ax.legend(loc='upper right')
    plt.savefig(f'{filename}{suffix}_I.png')
    plt.close()

    # Plot UR(f) and ULC(f)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(f, UR, 'g-', label=r'$U_R(f)$')
    ax.plot(f, ULC, 'm-', label=r'$U_{{LC}}(f)$')
    ax.set_xlabel(r'Frequenz $f$ [Hz]')
    ax.set_ylabel(r'Spannung [V]')
    ax.set_title(fr'Spannungsverläufe $U_R(f)$ und $U_{{LC}}(f)$ für {filename} {suffix}')
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.savefig(f'{filename}{suffix}_UR_ULC.png')
    plt.close()

    # Plot Ortskurve Z(ω)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    colors = cm.rainbow(np.linspace(0, 1, len(Z_real)))
    for i in range(len(Z_real)):
        if theoretical:
            label = r'$Z(\omega_1)$' if i == 0 else r'$Z(\omega_{end})$' if i == len(Z_real) - 1 else None
        else:
            label = fr'$Z(\omega_{{{i+1}}})$'
        if label:
            ax.plot([0, Z_real[i]], [0, Z_imag[i]], color=colors[i], marker='o', markersize=5, label=label)
        else:
            ax.plot([0, Z_real[i]], [0, Z_imag[i]], color=colors[i], marker='o', markersize=5)
    ax.set_xlabel(r'Reelle Achse [$\Omega$]')
    ax.set_ylabel(r'Imaginäre Achse [$\Omega$]')
    ax.set_title(fr'Ortskurve des komplexen Innenwiderstands $Z(\omega)$ für {filename} {suffix}')
    ax.grid(True)
    ax.legend(loc='upper right', ncol=2, fontsize='small')
    plt.savefig(f'{filename}{suffix}_Z_locus.png')
    plt.close()

    # Plot Ortskurve I(ω)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    colors = cm.rainbow(np.linspace(0, 1, len(I_real)))
    for i in range(len(I_real)):
        if theoretical:
            label = r'$I(\omega_1)$' if i == 0 else r'$I(\omega_{end})$' if i == len(I_real) - 1 else None
        else:
            label = fr'$I(\omega_{{{i+1}}})$'
        if label:
            ax.plot([0, I_real[i]], [0, I_imag[i]], color=colors[i], marker='o', markersize=5, label=label)
        else:
            ax.plot([0, I_real[i]], [0, I_imag[i]], color=colors[i], marker='o', markersize=5)
    ax.plot(I_real, I_imag, 'k--', label='Ortskurve (verbunden)')
    ax.set_xlabel(r'Re($I$) [mA]')
    ax.set_ylabel(r'Im($I$) [mA]')
    ax.set_title(fr'Ortskurve des komplexen Stromes $I(\omega)$ für {filename} {suffix}')
    ax.grid(True)
    ax.legend(loc='upper right', ncol=2, fontsize='small')
    plt.savefig(f'{filename}{suffix}_I_locus.png')
    plt.close()

    # Plot phi(f)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(f, phi_deg, 'r-', label=r'$\phi(f)$')
    ax.set_xlabel(r'Frequenz $f$ [Hz]')
    ax.set_ylabel(r'Phase $\phi$ [°]')
    ax.set_title(fr'Phasengang $\phi(f)$ für {filename} {suffix}')
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.savefig(f'{filename}{suffix}_phi.png')
    plt.close()

if __name__ == "__main__":
    root = Tk()
    root.withdraw()  # Hide the main window

    file1 = filedialog.askopenfilename(title="Wähle Datei für Kondensator 1")
    file2 = filedialog.askopenfilename(title="Wähle Datei für Kondensator 2")

    L_theo = simpledialog.askfloat("Spule L", "Gib L in H ein:")
    R_theo = simpledialog.askfloat("Widerstand R", "Gib R in Ohm ein:")
    C1_theo_uf = simpledialog.askfloat("Kondensator C1", "Gib C1 in µF ein:")
    C2_theo_uf = simpledialog.askfloat("Kondensator C2", "Gib C2 in µF ein:")
    U_theo = simpledialog.askfloat("Spannung U", "Gib U in V ein:")

    C1_theo = C1_theo_uf * 1e-6 if C1_theo_uf is not None else 0.47e-6
    C2_theo = C2_theo_uf * 1e-6 if C2_theo_uf is not None else 0.22e-6
    L_theo = L_theo if L_theo is not None else 0.1
    R_theo = R_theo if R_theo is not None else 100.0
    U_theo = U_theo if U_theo is not None else 1.0
    
    basename1 = os.path.splitext(os.path.basename(file1))[0].replace("csv ", "")
    basename2 = os.path.splitext(os.path.basename(file2))[0].replace("csv ", "")
    
    print(f"Ergebnisse für C1 (nominal {C1_theo_uf:.2f} µF):")
    R1, dR1, L1, dL1, C1, dC1, f01, Delta_f1, df1 = compute_params(file1)
    print(f"Resonanzfrequenz f0 = {f01:.2f} Hz")
    print(f"Bandbreite Δf = {Delta_f1:.2f} Hz")
    print(f"R = {R1:.2f} ± {dR1:.2f} Ω")
    print(f"L = {L1:.6f} ± {dL1:.6f} H")
    print(f"C1 = {C1 * 1e6:.2f} ± {dC1 * 1e6:.2f} μF")
    plot_data(df1, basename1, f01, Delta_f1, R1, L1, C1)
    
    print(f"\nErgebnisse für C2 (nominal {C2_theo_uf:.2f} µF):")
    R2, dR2, L2, dL2, C2, dC2, f02, Delta_f2, df2 = compute_params(file2)
    print(f"Resonanzfrequenz f0 = {f02:.2f} Hz")
    print(f"Bandbreite Δf = {Delta_f2:.2f} Hz")
    print(f"R = {R2:.2f} ± {dR2:.2f} Ω")
    print(f"L = {L2:.6f} ± {dL2:.6f} H")
    print(f"C2 = {C2 * 1e6:.2f} ± {dC2 * 1e6:.2f} μF")
    plot_data(df2, basename2, f02, Delta_f2, R2, L2, C2)
    
    f_min1 = df1['f / Hz'].min()
    f_max1 = df1['f / Hz'].max()
    df1_theo = generate_theoretical_df(f_min1, f_max1, 200, R_theo, L_theo, C1_theo, U_theo)
    f01_theo = 1 / (2 * np.pi * np.sqrt(L_theo * C1_theo))
    Delta_f1_theo = R_theo / (2 * np.pi * L_theo)
    plot_data(df1_theo, basename1, f01_theo, Delta_f1_theo, R_theo, L_theo, C1_theo, theoretical=True)
    
    f_min2 = df2['f / Hz'].min()
    f_max2 = df2['f / Hz'].max()
    df2_theo = generate_theoretical_df(f_min2, f_max2, 200, R_theo, L_theo, C2_theo, U_theo)
    f02_theo = 1 / (2 * np.pi * np.sqrt(L_theo * C2_theo))
    Delta_f2_theo = R_theo / (2 * np.pi * L_theo)
    plot_data(df2_theo, basename2, f02_theo, Delta_f2_theo, R_theo, L_theo, C2_theo, theoretical=True)