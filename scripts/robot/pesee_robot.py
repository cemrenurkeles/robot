"""
pesee.py - Balance à ressort : mesure d'une masse par la fréquence d'oscillation.

Module prévu pour être piloté par le programme du robot (LeRobot) :

    from pesee import Balance

    with Balance() as balance:          # ouvre la caméra OAK une seule fois
        ...                             # le robot pose le poids
        balance.demarrer()              # l'enregistrement démarre en arrière-plan
        ...                             # le robot appuie sur le plateau puis se retire
        resultat = balance.terminer()   # attend la fin du relevé, calcule la masse
        print(resultat["masse"])

Utilisation directe (sans robot), pour régler et tester :
    python pesee.py reglage               # régler les seuils HSV avec des curseurs
    python pesee.py test                  # relevé manuel puis calcul de la masse
    python pesee.py fichier positions.txt # recalcul à partir d'un relevé enregistré

Dépendances : pip install opencv-python numpy scipy matplotlib "depthai<3"
"""

import ast
import os
import sys
import threading

import cv2
import matplotlib
matplotlib.use("Agg")                   # graphes enregistrés en PNG, aucune fenêtre bloquante
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Sous Windows, la console n'est pas en UTF-8 par défaut (accents, ω, γ)
for _flux in (sys.stdout, sys.stderr):
    try:
        _flux.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


# =====================================================================
# CONFIGURATION
# =====================================================================

DOSSIER = os.path.dirname(os.path.abspath(__file__))

# --- Détection de la pastille verte ------------------------------------------
# Seuils HSV (H : 0-179, S et V : 0-255). Pour les régler : python pesee.py reglage
HSV_BAS = [47, 60, 90]
HSV_HAUT = [83, 255, 255]

MIN_AREA = 300          # aire minimale d'un candidat (px²)
MAX_ASPECT = 3.0        # rapport maximal grand côté / petit côté
MIN_FILL = 0.40         # remplissage minimal du rectangle englobant
CLOSE_SIZE = 15         # taille de la fermeture morphologique

# Zone de l'image où la pastille peut se trouver : (x1, y1, x2, y2) en pixels,
# ou None pour toute l'image. Évite les faux positifs (autre objet vert, bras).
ZONE = None

# --- Caméra ------------------------------------------------------------------
FPS = 60

# --- Relevé ------------------------------------------------------------------
DUREE = 8               # durée du relevé (s) : armement + oscillation complète
MIN_POINTS = 30         # nombre minimal de positions pour lancer le calcul

# --- Calcul ------------------------------------------------------------------
# Calibration : valeurs obtenues avec calibrer() (voir en bas du fichier)
K = 26463.16136712657   # raideur, en unité de masse · s⁻²
M0 = 21.182622376305517  # masse équivalente du plateau et du ressort

SKIP = 0.5              # secondes ignorées après le lâcher (même valeur qu'à la calibration !)
R2_MIN = 0.8            # en dessous, la mesure est signalée comme peu fiable

# --- Fichiers produits (à côté de ce script) ---------------------------------
FICHIER_POSITIONS = os.path.join(DOSSIER, "positions.txt")
FICHIER_CSV = os.path.join(DOSSIER, "mesure.csv")
FICHIER_COUPE = os.path.join(DOSSIER, "mesure_coupe.csv")
FICHIER_GRAPHE = os.path.join(DOSSIER, "ajustement.png")


# =====================================================================
# CAMÉRA OAK
# =====================================================================

class CameraOAK:
    """Caméra Luxonis OAK (DepthAI v2). Chaque image est horodatée par la caméra."""

    def __init__(self, fps=FPS):
        import depthai as dai

        pipeline = dai.Pipeline()
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setIspScale(2, 3)                       # 1920x1080 → 1280x720
        cam.setVideoSize(1280, 720)
        cam.setFps(fps)

        sortie = pipeline.create(dai.node.XLinkOut)
        sortie.setStreamName("rgb")
        cam.video.link(sortie.input)

        self.device = dai.Device(pipeline)
        self.queue = self.device.getOutputQueue("rgb", maxSize=8, blocking=False)

    def lire(self):
        """Renvoie (image, instant de prise de vue en secondes)."""
        img = self.queue.get()
        return img.getCvFrame(), img.getTimestamp().total_seconds()

    def vider(self):
        """Jette les images en attente, pour que le relevé parte d'images fraîches."""
        self.queue.tryGetAll()

    def fermer(self):
        self.device.close()


# =====================================================================
# DÉTECTION DE LA PASTILLE
# =====================================================================

def detecter(frame, hsv_bas=None, hsv_haut=None):
    """Cherche la pastille verte.

    Renvoie (position (x, y) ou None, masque, liste des candidats).
    Si plusieurs candidats, on garde le plus grand : pour écarter un autre objet
    vert de façon fiable, restreindre la recherche avec ZONE.
    """
    hsv_bas = HSV_BAS if hsv_bas is None else hsv_bas
    hsv_haut = HSV_HAUT if hsv_haut is None else hsv_haut

    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    masque = cv2.inRange(hsv, np.array(hsv_bas), np.array(hsv_haut))
    masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    noyau = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_SIZE, CLOSE_SIZE))
    masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, noyau)

    if ZONE is not None:
        x1, y1, x2, y2 = ZONE
        hors_zone = np.ones_like(masque, dtype=bool)
        hors_zone[y1:y2, x1:x2] = False
        masque[hors_zone] = 0

    contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidats = []
    for c in contours:
        enveloppe = cv2.convexHull(c)
        aire = cv2.contourArea(enveloppe)
        if aire < MIN_AREA:
            continue
        (cx, cy), (w, h), _ = cv2.minAreaRect(enveloppe)
        if min(w, h) < 1 or max(w, h) / min(w, h) > MAX_ASPECT or aire / (w * h) < MIN_FILL:
            continue
        candidats.append((cx, cy, aire))

    if not candidats:
        return None, masque, candidats

    cx, cy, _ = max(candidats, key=lambda c: c[2])
    return (float(cx), float(cy)), masque, candidats


# =====================================================================
# BALANCE : relevé en arrière-plan, puis calcul
# =====================================================================

class Balance:
    """Pilote la caméra et le calcul. Utilisable depuis n'importe quel programme."""

    def __init__(self, camera=None):
        print("Ouverture de la caméra…")
        self.camera = camera if camera is not None else CameraOAK(FPS)
        self.positions = []
        self._fil = None
        self._stop = threading.Event()
        self._erreur = None
        print("Caméra prête.")

    # --- Relevé ----------------------------------------------------------------

    def demarrer(self, duree=DUREE):
        """Lance le relevé en arrière-plan et rend la main tout de suite."""
        if self._fil is not None and self._fil.is_alive():
            raise RuntimeError("un relevé est déjà en cours")
        self.positions = []
        self.nb_images = 0
        self._erreur = None
        self._stop.clear()
        self.camera.vider()
        self._fil = threading.Thread(target=self._boucle, args=(duree,), daemon=True)
        self._fil.start()

    def _boucle(self, duree):
        try:
            t0 = None
            while not self._stop.is_set():
                frame, instant = self.camera.lire()
                t0 = instant if t0 is None else t0
                t = instant - t0
                if t >= duree:
                    break
                self.nb_images += 1
                position, _, _ = detecter(frame)
                if position is not None:
                    self.positions.append([round(t, 4), round(position[0], 1), round(position[1], 1)])
        except Exception as erreur:          # transmise au programme principal par terminer()
            self._erreur = erreur

    def est_termine(self):
        """True quand le relevé est fini (non bloquant, pour une boucle de contrôle)."""
        return self._fil is not None and not self._fil.is_alive()

    def arreter(self):
        """Interrompt le relevé avant la fin prévue."""
        self._stop.set()

    def terminer(self, enregistrer=True):
        """Attend la fin du relevé, puis calcule la masse. Renvoie un dictionnaire."""
        if self._fil is None:
            raise RuntimeError("aucun relevé lancé : appeler demarrer() d'abord")
        self._fil.join()
        if self._erreur is not None:
            raise RuntimeError(f"erreur pendant le relevé : {self._erreur}") from self._erreur

        duree_reelle = self.positions[-1][0] if self.positions else 0
        print(f"Relevé terminé : {self.nb_images} images, "
              f"{len(self.positions)} détections"
              + (f", {self.nb_images / duree_reelle:.0f} images/s" if duree_reelle > 0 else ""))

        if enregistrer:
            with open(FICHIER_POSITIONS, "w", encoding="utf-8") as f:
                f.write(str(self.positions))

        return calcul(self.positions)

    def mesurer(self, duree=DUREE):
        """Relevé complet bloquant (démarrer + terminer)."""
        self.demarrer(duree)
        return self.terminer()

    # --- Fermeture ---------------------------------------------------------------

    def fermer(self):
        self.arreter()
        if self._fil is not None:
            self._fil.join(timeout=2)
        self.camera.fermer()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.fermer()


# =====================================================================
# CALCUL : des positions jusqu'à la masse
# =====================================================================

def charger(chemin):
    """Lit un relevé : liste [[t, x, y], ...] (txt) ou CSV t,x,y."""
    with open(chemin, encoding="utf-8") as f:
        texte = f.read().strip()
    if texte.startswith("["):
        return ast.literal_eval(texte)
    return np.loadtxt(chemin, delimiter=",", skiprows=1).tolist()


def ecrire_csv(chemin, d):
    np.savetxt(chemin, d, delimiter=",", header="t,x,y", comments="", fmt="%.4f")


def extremums(s, h):
    """Indices des extremums successifs (alternance max/min) séparés d'au moins h."""
    ext, tendance, i_min, i_max, cand = [], 0, 0, 0, 0
    for i in range(1, len(s)):
        if tendance == 0:
            i_min = i if s[i] < s[i_min] else i_min
            i_max = i if s[i] > s[i_max] else i_max
            if s[i] - s[i_min] > h:
                ext, tendance, cand = [i_min], 1, i
            elif s[i_max] - s[i] > h:
                ext, tendance, cand = [i_max], -1, i
        elif tendance * (s[i] - s[cand]) > 0:
            cand = i                                    # l'extremum continue de s'étendre
        elif tendance * (s[cand] - s[i]) > h:
            ext.append(cand)                            # retournement confirmé
            tendance, cand = -tendance, i
    return ext


def trouver_oscillation(t, x, y, fenetre=0.3):
    """Indices (début, fin) de l'oscillation : du lâcher au retour au repos."""
    pts = np.column_stack([x - x.mean(), y - y.mean()])
    _, vecteurs = np.linalg.eigh(np.cov(pts.T))
    s = pts @ vecteurs[:, -1]
    if np.ptp(s) < 20:
        raise RuntimeError("aucun mouvement détecté : la pastille est restée quasi immobile")

    # Sommets successifs : chaque « jambe » relie un sommet au suivant
    e = extremums(s, 0.15 * np.ptp(s))
    if len(e) < 4:
        raise RuntimeError("pas d'oscillation détectée")
    ampl = np.abs(np.diff(s[e]))

    # Départ de chaque jambe : dernier point encore proche du sommet (maintien avant lâcher)
    depart = [a + int(np.where(np.abs(s[a:b] - s[a]) < 0.1 * ampl[j])[0][-1])
              for j, (a, b) in enumerate(zip(e[:-1], e[1:]))]
    duree = t[e[1:]] - t[depart]
    demi_periode = np.median(duree)

    # On remonte depuis la fin tant que les jambes ressemblent à une oscillation
    # (durée ≈ demi-période, amplitude qui décroît doucement). L'armement casse la série.
    k = len(duree) - 1
    while k > 0 and duree[k - 1] < 2 * demi_periode and ampl[k - 1] > 0.7 * ampl[k]:
        k -= 1
    debut = depart[k]

    # Fin : dernière image encore agitée
    agitation = np.array([np.hypot(x[(t > ti - fenetre) & (t <= ti)].std(),
                                   y[(t > ti - fenetre) & (t <= ti)].std()) for ti in t])
    actifs = np.where(agitation[debut:] > 0.08 * agitation[debut:].max())[0]
    return debut, debut + int(actifs[-1])


def preparer(t, x, y, skip=SKIP):
    """Ignore les `skip` premières secondes, projette sur la direction du mouvement."""
    garde = t >= t[0] + skip
    t, x, y = t[garde], x[garde], y[garde]
    pts = np.column_stack([x - x.mean(), y - y.mean()])
    _, vecteurs = np.linalg.eigh(np.cov(pts.T))
    s = pts @ vecteurs[:, -1]
    return t - t[0], s - s.mean()


def frequence_fft(t, s):
    """Estimation grossière de la fréquence (point de départ de l'ajustement)."""
    tu = np.linspace(t[0], t[-1], len(t))
    su = np.interp(tu, t, s)
    spectre = np.abs(np.fft.rfft(su))
    freqs = np.fft.rfftfreq(len(tu), tu[1] - tu[0])
    return freqs[np.argmax(spectre[1:]) + 1]


def sinus_amorti(t, A, g, w, phi, c):
    return A * np.exp(-g * t) * np.cos(w * t + phi) + c


def ajuster(t, s, f_guess):
    p0 = [np.ptp(s) / 2, 0.1, 2 * np.pi * f_guess, 0, 0]
    params, _ = curve_fit(sinus_amorti, t, s, p0=p0, maxfev=10000)
    A, g, w, phi, c = params
    if A < 0:
        A, phi = -A, phi + np.pi
    g = abs(g)
    params = (A, g, w, phi, c)
    r2 = 1 - np.var(s - sinus_amorti(t, *params)) / np.var(s)
    return np.sqrt(w**2 + g**2), params, r2


def masse(w0, k=K, m0=M0):
    return k / w0**2 - m0


def tracer(t, s, params, chemin=FICHIER_GRAPHE, titre=""):
    tf = np.linspace(t[0], t[-1], 2000)
    plt.figure(figsize=(9, 4.5))
    plt.plot(t, s, ".", ms=4, label="mesures")
    plt.plot(tf, sinus_amorti(tf, *params), "-", lw=1.2, label="ajustement")
    plt.xlabel("temps (s)")
    plt.ylabel("position (px)")
    plt.title(titre)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(chemin, dpi=110)
    plt.close()


def calcul(positions, enregistrer=True):
    """Des positions [[t, x, y], ...] jusqu'à la masse. Renvoie un dictionnaire.

    Lève RuntimeError si la mesure est impossible (trop peu de points, pas d'oscillation).
    """
    d = np.asarray(positions, dtype=float)
    if len(d) < MIN_POINTS:
        raise RuntimeError(f"seulement {len(d)} positions détectées (minimum {MIN_POINTS})")
    t, x, y = d[:, 0], d[:, 1], d[:, 2]

    i0, i1 = trouver_oscillation(t, x, y)
    coupe = d[i0:i1 + 1]
    if enregistrer:
        ecrire_csv(FICHIER_CSV, d)
        ecrire_csv(FICHIER_COUPE, coupe)

    tc, s = preparer(coupe[:, 0], coupe[:, 1], coupe[:, 2])
    if len(tc) < 10:
        raise RuntimeError("oscillation trop courte après découpe")
    f_guess = frequence_fft(tc, s)
    w0, params, r2 = ajuster(tc, s, f_guess)
    m = masse(w0)

    resultat = {
        "masse": float(m),
        "w0": float(w0),
        "frequence": float(w0 / (2 * np.pi)),
        "amortissement": float(params[1]),
        "r2": float(r2),
        "fiable": bool(r2 >= R2_MIN),
        "t_lacher": float(t[i0]),
        "t_fin": float(t[i1]),
        "nb_points": len(tc),
    }

    if enregistrer:
        tracer(tc, s, params, titre=f"ω0 = {w0:.3f} rad/s, R² = {r2:.3f}, masse = {m:.2f}")

    print(f"Oscillation de t = {t[i0]:.3f} s à t = {t[i1]:.3f} s ({len(tc)} points)")
    print(f"Pulsation propre ω0 : {w0:.4f} rad/s   (γ = {params[1]:.3f} /s, R² = {r2:.3f})")
    print(f"Masse estimée       : {m:.2f}" + ("" if resultat["fiable"] else "   ⚠ PEU FIABLE"))
    return resultat


def calibrer(masses, w0s):
    """Masses connues et ω0 mesurés → (k, m0), via la droite 1/ω0² = (m + m0)/k."""
    a, b = np.polyfit(np.array(masses), 1 / np.array(w0s) ** 2, 1)
    return 1 / a, b / a


# =====================================================================
# OUTILS DE RÉGLAGE ET DE TEST (hors robot)
# =====================================================================

def reglage_hsv(camera):
    """Fenêtre avec curseurs pour régler les seuils HSV. q pour quitter."""
    fen = "Reglage HSV (q pour quitter)"
    cv2.namedWindow(fen)
    for nom, val, maxi in [("H min", HSV_BAS[0], 179), ("H max", HSV_HAUT[0], 179),
                           ("S min", HSV_BAS[1], 255), ("V min", HSV_BAS[2], 255)]:
        cv2.createTrackbar(nom, fen, val, maxi, lambda _: None)

    while True:
        frame, _ = camera.lire()
        bas = [cv2.getTrackbarPos("H min", fen), cv2.getTrackbarPos("S min", fen),
               cv2.getTrackbarPos("V min", fen)]
        haut = [cv2.getTrackbarPos("H max", fen), 255, 255]
        position, masque, candidats = detecter(frame, bas, haut)

        vue = frame.copy()
        if ZONE is not None:
            cv2.rectangle(vue, ZONE[:2], ZONE[2:], (255, 0, 0), 2)
        for cx, cy, _ in candidats:
            cv2.circle(vue, (int(cx), int(cy)), 8, (0, 255, 255), 2)
        if position is not None:
            cv2.drawMarker(vue, tuple(map(int, position)), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        both = np.hstack([vue, cv2.cvtColor(masque, cv2.COLOR_GRAY2BGR)])
        cv2.imshow(fen, cv2.resize(both, None, fx=0.5, fy=0.5))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("\nÀ reporter dans la configuration de pesee.py :")
    print(f"HSV_BAS = {bas}")
    print(f"HSV_HAUT = {haut}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    if mode == "fichier":
        calcul(charger(sys.argv[2]))
        print(f"Graphe : {FICHIER_GRAPHE}")

    elif mode == "reglage":
        cam = CameraOAK(FPS)
        try:
            reglage_hsv(cam)
        finally:
            cam.fermer()

    elif mode == "test":
        with Balance() as balance:
            input("Pose le poids, puis appuie sur Entrée et lance l'oscillation à la main… ")
            balance.demarrer()
            print(f"Relevé en cours ({DUREE} s)…")
            try:
                balance.terminer()
                print(f"Graphe : {FICHIER_GRAPHE}")
            except RuntimeError as erreur:
                print(f"Mesure impossible : {erreur}")

    else:
        print(__doc__)
