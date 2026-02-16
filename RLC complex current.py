import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tkinter as tk
from tkinter import simpledialog, filedialog
import os

def load_measured_data(filename):
    """
    Lädt die Messdaten aus einer CSV-Datei.
    Erwartet Spalten wie 'f / Hz', 'I / mA' und eine Phase 'phi'.
    """
    try:
        df = pd.read_csv(filename, encoding='latin1')
        
        # Sicherstellen, dass nach Frequenz sortiert ist
        if 'f / Hz' in df.columns:
            df = df.sort_values(by='f / Hz').reset_index(drop=True)
            f_min = df['f / Hz'].min()
            f_max = df['f / Hz'].max()
            return df, f_min, f_max
        else:
            print(f"Fehler: Keine 'f / Hz' Spalte in {filename} gefunden.")
            return None, None, None
            
    except Exception as e:
        print(f"Fehler beim Laden von {filename}: {e}")
        return None, None, None

def get_user_input():
    root = tk.Tk()
    root.withdraw()  # Hauptfenster verbergen

    def ask(title, prompt, initial, cast=float):
        try:
            val = simpledialog.askstring(title, prompt, initialvalue=str(initial))
            if val is None: return initial
            return cast(val)
        except ValueError:
            return initial

    # Abfrage: Daten laden?
    load_data_str = simpledialog.askstring("Daten laden", "Möchten Sie Messdaten laden? (y/n)", initialvalue="n")
    load_data = load_data_str is not None and load_data_str.lower().strip() == 'y'

    circuit_type = ask("Schaltungstyp", "Schaltung ('seriel' oder 'parallel'):", "seriel", str).lower()
    if circuit_type not in ['seriel', 'parallel']:
        circuit_type = 'seriel'

    # Gemeinsame Parameter
    R = ask("Parameter", "Widerstand R (Ohm):", 100.0)
    L_mH = ask("Parameter", "Induktivität L (mH):", 10.0)
    U_gen = ask("Parameter", "Generator Spannung U_gen (V):", 1.0)
    Ri = ask("Parameter", "Innenwiderstand Ri (Ohm) [default 0]:", 0.0)

    configs = []

    if load_data:
        # Fall A: Messdaten laden (2 Dateien)
        print("Bitte wählen Sie die CSV-Dateien aus.")
        
        # Konfiguration 1
        C1_uF = ask("Kondensator 1", "Kapazität C1 (µF):", 1.0)
        file1 = filedialog.askopenfilename(title="Wähle CSV Datei für C1")
        if file1:
            configs.append({'C_uF': C1_uF, 'filename': file1, 'manual_freq': False})
            
        # Konfiguration 2
        C2_uF = ask("Kondensator 2", "Kapazität C2 (µF):", 0.47)
        file2 = filedialog.askopenfilename(title="Wähle CSV Datei für C2")
        if file2:
            configs.append({'C_uF': C2_uF, 'filename': file2, 'manual_freq': False})
            
    else:
        # Fall B: Nur Theorie
        C_uF = ask("Parameter", "Kapazität C (µF):", 1.0)
        f_min = ask("Frequenz", "Startfrequenz f_min (Hz):", 10.0)
        f_max = ask("Frequenz", "Endfrequenz f_max (Hz):", 50000.0)
        num_points = ask("Frequenz", "Anzahl der Punkte:", 200, int)
        
        configs.append({
            'C_uF': C_uF, 
            'filename': None, 
            'manual_freq': True,
            'f_min': f_min,
            'f_max': f_max,
            'num_points': num_points
        })

    root.destroy()
    return circuit_type, R, L_mH * 1e-3, U_gen, Ri, configs

def generate_theoretical_data(ctype, R, L, C, U_gen, f_min, f_max, num_points, Ri):
    # Resonanzfrequenz
    f0 = 1 / (2 * np.pi * np.sqrt(L * C))

    # Frequenz-Array erstellen (Log + Lin um Resonanz)
    if num_points < 10: num_points = 200
    
    f_log = np.logspace(np.log10(max(f_min, 1e-1)), np.log10(f_max), num_points // 2)
    span = f0 * 0.5
    f_lin = np.linspace(max(f_min, f0 - span), min(f_max, f0 + span), num_points // 2)
    f = np.sort(np.unique(np.concatenate((f_log, f_lin))))
    w = 2 * np.pi * f

    if ctype == 'seriel':
        # Serien-RLC
        # Z_RLC = R + j(wL - 1/wC)
        # Z_total = Ri + Z_RLC
        X = w * L - 1 / (w * C)
        Z_total = (R + Ri) + 1j * X
        
        with np.errstate(divide='ignore', invalid='ignore'):
            I = U_gen / Z_total
            
        Delta_f = (R + Ri) / (2 * np.pi * L)

    else:
        # Parallel-RLC
        # Y = 1/R + 1/(jwL) + jwC
        # Z = 1/Y
        with np.errstate(divide='ignore', invalid='ignore'):
            Y = 1/R + 1/(1j * w * L) + 1j * w * C
            Z = 1 / Y
        
        if Ri > 0:
            Z_total = Ri + Z
            with np.errstate(divide='ignore', invalid='ignore'):
                I = U_gen / Z_total
        else:
            Z_total = Z
            with np.errstate(divide='ignore', invalid='ignore'):
                I = U_gen / Z
        
        Delta_f = 1 / (2 * np.pi * R * C)

    # Z für Plot (Impedanz der Last, ohne Ri, oder Gesamt? Üblich: Lastimpedanz Z(w))
    # Wir nehmen hier Z_total als "gesamte Last aus Sicht der quelle U_gen"
    # Wenn Ri=0, ist Z_total = Z_RLC
    
    return f, Z_total, I, f0, Delta_f

def process_measured_data(df, U_gen):
    """
    Berechnet komplexe Z und I aus Messdaten.
    """
    f = df['f / Hz'].values
    I_ma = df['I / mA'].values
    I_amp = I_ma / 1000.0
    
    # Phase suchen
    phi_col = [col for col in df.columns if 'phi' in col.lower()]
    if phi_col:
        phi_deg = df[phi_col[0]].values
    else:
        phi_deg = np.zeros_like(I_amp)
        
    phi_rad = np.deg2rad(phi_deg)
    
    # I komplex: Betrag I_amp, Phase -phi (sofern phi die Phasenverschiebung U zu I ist)
    # Wenn U = U0 * e^0, I = (U0/Z) * e^-j*phi_z
    I_real = I_ma * np.cos(-phi_rad)
    I_imag = I_ma * np.sin(-phi_rad)
    I_complex = (I_real + 1j * I_imag) / 1000.0 # in A

    # Z komplex: U_gen / I_complex
    # Achtung: Wenn I sehr klein, Z riesig
    with np.errstate(divide='ignore', invalid='ignore'):
        Z_complex = U_gen / I_complex

    return f, Z_complex, I_complex * 1000.0 # I wieder in mA für Konsistenz? Nein hier A, draußen mA

def plot_results(ctype, cfg_theo, cfg_exp=None):
    """
    Erstellt alle Diagramme: Ortskurven und Frequenzgänge.
    cfg_theo: Dictionary mit Theorie-Daten (f, Z, I, f0, Df, R, L, C, U_gen, Ri)
    cfg_exp: Dictionary mit Messdaten (df_meas) - Optional
    """
    f_theo = cfg_theo['f']
    Z_theo = cfg_theo['Z']
    I_theo = cfg_theo['I']
    f0_theo = cfg_theo['f0']
    R_nom = cfg_theo['R']
    L_nom = cfg_theo['L']
    C_nom = cfg_theo['C']
    U_gen = cfg_theo['U_gen']
    
    df_meas = cfg_exp['df_meas'] if cfg_exp else None
    
    # Einstellungen
    fig_width = 19.2
    fig_height = 10.8
    title_suffix = f" ({ctype.capitalize()} RLC: R={R_nom}Ω, L={L_nom*1e3:.1f}mH, C={C_nom*1e6:.1f}µF)"
    file_suffix = f"_C_{C_nom*1e6:.1f}uF"

    # Messdaten verarbeiten falls vorhanden
    if df_meas is not None:
        f_meas, Z_meas, I_meas_A = process_measured_data(df_meas, U_gen)
        I_meas_mA = I_meas_A * 1000.0
        
        # UR und ULC aus Messdaten extrahieren
        # ULC ist oft U - UR oder direkt gemessen
        UR_meas = df_meas['UR / V'].values if 'UR / V' in df_meas.columns else np.abs(I_meas_A) * R_nom
        ULC_meas = df_meas['ULC / V'].values if 'ULC / V' in df_meas.columns else np.abs(I_meas_A) * np.abs(Z_meas - R_nom)
        
        # Phase phi aus Messdaten
        phi_col = [col for col in df_meas.columns if 'phi' in col.lower()]
        phi_meas = df_meas[phi_col[0]].values if phi_col else np.angle(Z_meas, deg=True)
    
    # Theorie-Werte für Frequenzgänge
    w_theo = 2 * np.pi * f_theo
    X_theo = w_theo * L_nom - 1 / (w_theo * C_nom)
    UR_theo = np.abs(I_theo) * R_nom
    ULC_theo = np.abs(I_theo) * np.abs(X_theo)
    phi_theo = np.angle(Z_theo, deg=True)
    Z_mag_theo = np.abs(Z_theo)

    # --- 1. Ortskurve Z ---
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(np.real(Z_theo), np.imag(Z_theo), 'k-', alpha=0.5, label='Theorie (Linie)')
    if df_meas is not None:
        ax.scatter(np.real(Z_meas), np.imag(Z_meas), c=f_meas, cmap='viridis', label='Messung (Punkte)', zorder=10)
        ax.plot(np.real(Z_meas), np.imag(Z_meas), 'k:', alpha=0.4, zorder=9)
    ax.set_xlabel(r'Re($Z$) [$\Omega$]')
    ax.set_ylabel(r'Im($Z$) [$\Omega$]')
    ax.set_title(f'Ortskurve der Impedanz Z{title_suffix}')
    ax.grid(True)
    ax.legend()
    plt.savefig(f"{ctype}_Z{file_suffix}.png")
    plt.close()

    # --- 2. Ortskurve I ---
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(np.real(I_theo*1000), np.imag(I_theo*1000), 'k--', label='Theorie (Linie)')
    if df_meas is not None:
        ax.scatter(np.real(I_meas_mA), np.imag(I_meas_mA), c=f_meas, cmap='viridis', label='Messung (Punkte)', zorder=10)
        ax.plot(np.real(I_meas_mA), np.imag(I_meas_mA), 'r:', alpha=0.4)
    ax.set_xlabel(r'Re($I$) [mA]')
    ax.set_ylabel(r'Im($I$) [mA]')
    ax.set_title(f'Ortskurve des Stromes I{title_suffix}')
    ax.grid(True)
    ax.legend()
    plt.savefig(f"{ctype}_I{file_suffix}.png")
    plt.close()

    # --- 3. I(f) Stromkennlinie ---
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(f_theo, np.abs(I_theo)*1000, 'b-', label='Theorie')
    if df_meas is not None:
        ax.plot(f_meas, np.abs(I_meas_mA), 'ro', markersize=4, label='Messung')
        ax.plot(f_meas, np.abs(I_meas_mA), 'r:', alpha=0.5)
    ax.set_xlabel('Frequenz f [Hz]')
    ax.set_ylabel('Strom I [mA]')
    ax.set_title(f'Stromkennlinie I(f){title_suffix}')
    ax.grid(True)
    ax.legend()
    plt.savefig(f"{ctype}_I_freq{file_suffix}.png")
    plt.close()

    # --- 4. UR(f) & ULC(f) ---
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(f_theo, UR_theo, 'g-', label=r'$U_R$ Theorie')
    ax.plot(f_theo, ULC_theo, 'm-', label=r'$U_{LC}$ Theorie')
    if df_meas is not None:
        ax.plot(f_meas, UR_meas, 'go', markersize=4, alpha=0.6, label=r'$U_R$ Messung')
        ax.plot(f_meas, ULC_meas, 'mo', markersize=4, alpha=0.6, label=r'$U_{LC}$ Messung')
    ax.set_xlabel('Frequenz f [Hz]')
    ax.set_ylabel('Spannung U [V]')
    ax.set_title(f'Spannungsverläufe UR(f) und ULC(f){title_suffix}')
    ax.grid(True)
    ax.legend()
    plt.savefig(f"{ctype}_voltages{file_suffix}.png")
    plt.close()

    # --- 5. phi(f) Phasenwinkel ---
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(f_theo, phi_theo, 'k-', label='Theorie')
    if df_meas is not None:
        ax.plot(f_meas, phi_meas, 'bo', markersize=4, label='Messung')
        ax.plot(f_meas, phi_meas, 'b:', alpha=0.5)
    ax.set_xlabel('Frequenz f [Hz]')
    ax.set_ylabel(r'Phase $\phi$ [°]')
    ax.set_title(f'Phasengang phi(f){title_suffix}')
    ax.grid(True)
    ax.legend()
    plt.savefig(f"{ctype}_phi_freq{file_suffix}.png")
    plt.close()

    # --- 6. |Z|(f) Impedanzverlauf ---
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(f_theo, Z_mag_theo, 'b-', label='Theorie')
    if df_meas is not None:
        ax.plot(f_meas, np.abs(Z_meas), 'ro', markersize=4, label='Messung')
        ax.plot(f_meas, np.abs(Z_meas), 'r:', alpha=0.5)
    ax.set_xlabel('Frequenz f [Hz]')
    ax.set_ylabel(r'Impedanz $|Z|$ [$\Omega$]')
    ax.set_title(f'Verlauf der Impedanz |Z|(f){title_suffix}')
    ax.grid(True)
    ax.legend()
    plt.savefig(f"{ctype}_Z_mag_freq{file_suffix}.png")
    plt.close()

def compute_params(df, Ri=0):
    """
    Berechnet R, L, C mit Gauß'scher Fehlerfortpflanzung aus den Messdaten.
    Logik übernommen aus RLC Datenauswertung.py.
    """
    f = df['f / Hz'].values
    I_ma = df['I / mA'].values
    
    # phi Spalte suchen
    phi_col = [col for col in df.columns if 'phi' in col.lower()][0]
    phi_deg = df[phi_col].values
    
    # UR und ULC falls vorhanden, sonst schätzen?
    # RLC Datenauswertung nutzt UR und ULC direkt aus CSV.
    if 'UR / V' in df.columns and 'ULC / V' in df.columns:
        UR = df['UR / V'].values
    else:
        # Fallback falls Spalten fehlen
        print("Warnung: 'UR / V' nicht gefunden. Schätze UR = I * R_nominal...")
        UR = (I_ma / 1000.0) * 100.0 # Sehr grob

    I_amp = I_ma / 1000.0

    # Resonanz suchen: max I
    idx_res = np.argmax(I_amp)
    f0 = f[idx_res]
    I_max = I_amp[idx_res]

    # Fehler f0 schätzen
    if idx_res > 0 and idx_res < len(f) - 1:
        delta_f0 = 0.5 * min(f[idx_res] - f[idx_res-1], f[idx_res+1] - f[idx_res])
    else:
        delta_f0 = 1.0

    # R berechnen (Mittelwert UR / I)
    valid_mask = I_amp != 0
    R_values = UR[valid_mask] / I_amp[valid_mask]
    R = np.mean(R_values)
    delta_R = np.std(R_values, ddof=1) / np.sqrt(len(R_values)) if len(R_values) > 1 else 1.0

    # Bandbreite suchen (I_max / sqrt(2))
    I_target = I_max / np.sqrt(2)
    
    # Drunter
    lower_mask = f < f0
    if np.any(lower_mask):
        f_lower = f[lower_mask]
        I_lower = I_amp[lower_mask]
        idx_lower = np.argmin(np.abs(I_lower - I_target))
        f1 = f_lower[idx_lower]
        idx_l_global = np.where(f == f1)[0][0]
        delta_f1 = 0.5 * (f[idx_l_global+1] - f[idx_l_global-1]) if 0 < idx_l_global < len(f)-1 else 1.0
    else:
        f1, delta_f1 = f0, 1.0

    # Drüber
    upper_mask = f > f0
    if np.any(upper_mask):
        f_upper = f[upper_mask]
        I_upper = I_amp[upper_mask]
        idx_upper = np.argmin(np.abs(I_upper - I_target))
        f2 = f_upper[idx_upper]
        idx_u_global = np.where(f == f2)[0][0]
        delta_f2 = 0.5 * (f[idx_u_global+1] - f[idx_u_global-1]) if 0 < idx_u_global < len(f)-1 else 1.0
    else:
        f2, delta_f2 = f0, 1.0

    Delta_f = f2 - f1
    delta_Delta_f = np.sqrt(delta_f1**2 + delta_f2**2)

    # L und C berechnen (Serien-Formeln)
    R_eff = R + Ri
    L = R_eff / (2 * np.pi * Delta_f)
    delta_L = L * np.sqrt((delta_R / R_eff)**2 + (delta_Delta_f / Delta_f)**2)

    C = 1 / (4 * np.pi**2 * f0**2 * L)
    delta_C = C * np.sqrt((delta_L / L)**2 + (2 * delta_f0 / f0)**2)

    return R, delta_R, L, delta_L, C, delta_C, f0, Delta_f

if __name__ == "__main__":
    # 1. User Input
    ctype, R_nom, L_nom, U_gen, Ri, configs = get_user_input()
    
    # 2. Loop durch Konfigurationen
    for cfg in configs:
        C_nom = cfg['C_uF'] * 1e-6
        filename = cfg['filename']
        df_meas = None
        
        if filename:
            print(f"\n" + "="*50)
            print(f"VERARBEITE DATEI: {os.path.basename(filename)}")
            print(f"Nominale Kapazität: {cfg['C_uF']:.2f} µF")
            
            df_meas, f_min_meas, f_max_meas = load_measured_data(filename)
            if df_meas is not None:
                # Parameter berechnen
                R_exp, dR_exp, L_exp, dL_exp, C_exp, dC_exp, f0_exp, Df_exp = compute_params(df_meas, Ri)
                
                print(f"\nERGEBNISSE (Experimentell):")
                print(f"  Resonanzfrequenz f0 = {f0_exp:.2f} Hz")
                print(f"  Bandbreite Δf = {Df_exp:.2f} Hz")
                print(f"  Widerstand R = {R_exp:.6f} ± {dR_exp:.6f} Ω")
                print(f"  Induktivität L = {L_exp:.6f} ± {dL_exp:.6f} H")
                print(f"  Kapazität C = {C_exp*1e6:.6f} ± {dC_exp*1e6:.6f} µF")
                
                f_min, f_max = f_min_meas, f_max_meas
                num_points = 200
                
                # Für Theorie-Plot nehmen wir die EXPERIMENTELLEN Werte als Basis? 
                # Oder die nominalen? Datenauswertung.py nimmt nominale für Theorie-DF.
                # Wir bleiben bei nominalen für "Theorie", aber plotten Messung drüber.
            else:
                f_min, f_max, num_points = 10, 50000, 200
        else:
            f_min = cfg['f_min']
            f_max = cfg['f_max']
            num_points = cfg['num_points']

        # 3. Theorie Berechnen
        # Wir nutzen hier die nominalen Werte vom User-Input am Anfang
        f_theo, Z_theo, I_theo, f0_theo, Df_theo = generate_theoretical_data(
            ctype, R_nom, L_nom, C_nom, U_gen, f_min, f_max, num_points, Ri
        )
        
        # 4. Daten für Plotting vorbereiten
        cfg_theo = {
            'f': f_theo,
            'Z': Z_theo,
            'I': I_theo,
            'f0': f0_theo,
            'Df': Df_theo,
            'R': R_nom,
            'L': L_nom,
            'C': C_nom,
            'U_gen': U_gen,
            'Ri': Ri
        }
        cfg_exp = {'df_meas': df_meas} if df_meas is not None else None

        # 5. Plotten
        plot_results(ctype, cfg_theo, cfg_exp)
    
    print("\n" + "="*50)
    print("Fertig. Alle Berechnungen und Diagramme erstellt.")