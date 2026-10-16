import sys
import os
import time
import math
import json
import random
from collections import deque

import numpy as np
import cv2

try:
    import sounddevice as sd
except Exception: 
    sd = None

try:
    import mediapipe as mp
except Exception:  # يوجد بديل احتياطي (Haar) إذا لم تتوفر المكتبة
    mp = None

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QFrame, QProgressBar, QStackedWidget,
    QGraphicsDropShadowEffect, QSizePolicy
)


APP_TITLE = "مرآة الأكاذيب الذكية"
ROUNDS = 5                      
SCAN_SECONDS = 4               
ROUND_ANALYSIS_SECONDS = 2.0    
FINAL_ANALYSIS_SECONDS = 5.0    
AUTO_RESET_SECONDS = 30        
MIN_FACE_COVERAGE = 0.5        

QUESTIONS = [
    "عمرك قلت انك بالطريق وانت بعدك ع التخت ",
    "عمره حدا شرحلك فكرة معينة واعطيته ابتسامة انك فاهم وانت ما فهمت اشي؟",
    "عمرك انطردت من محاضرة؟واذا اه شو السبب؟",
    "هل بيوم من الايام الوادة قالتلك انتبه ع الاكل واجتك لحظة ادراك بعد ساعتين وحرقتها؟",
    "بتخاف من العتمة؟",
    "هل سيناريو الهوشة اللي تخيلته بالشاور كان أحسن من الهوشة الحقيقية؟",
    "عمرك سويت حالك مشغول بالموبايل عشان تتفادى تسلم على حد؟",
    "هل سبق وقلت للدكتور 'الفكرة ببالي بس مش عارف أعبرعنها وأنت مش فاهم إشي؟",
    "بتحب دفعتك؟",
    "كم مرة سحبت باب مكتوب عليه Push؟",
    "عمرك كنت تحكي عن شخص ولقيته وراك ؟",
]

def load_questions():
    """يقرأ الأسئلة من questions.txt (سؤال في كل سطر، والسطر الذي يبدأ بـ # يُتجاهل).
    إذا لم يوجد الملف أو قلّت أسئلته عن عدد الجولات تُستخدم القائمة الافتراضية أعلاه."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "questions.txt")
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            qs = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        qs = list(dict.fromkeys(qs))  # إزالة المكرر مع حفظ الترتيب
        if len(qs) >= ROUNDS:
            return qs
    except Exception:
        pass
    return list(QUESTIONS)


class QuestionDeck:
    """يسحب الأسئلة كأنها طقم أوراق: لا يتكرر سؤال قبل أن تُستخدم كل الأسئلة،
    فيحصل كل زائر جديد على أسئلة مختلفة عن الزائر السابق قدر الإمكان."""

    def __init__(self, questions):
        self.all = list(questions)
        self.deck = []

    def draw(self, n):
        n = min(n, len(self.all))
        picked = []
        while len(picked) < n:
            if not self.deck:
                self.deck = [q for q in self.all if q not in picked]
                random.shuffle(self.deck)
            picked.append(self.deck.pop())
        return picked


ANALYSIS_MESSAGES = [
    "جاري تحليل البيانات العصبية والحركات الدقيقة...",
    "معالجة الأنماط السلوكية...",
    "تحليل نبرة الصوت وإيقاع الكلام...",
    "مطابقة معالم الوجه مع النموذج المرجعي...",
    "حساب معدل الانحراف الحركي (Movement Delta)...",
    "دمج المؤشرات وتحديد نسبة الثقة...",
    "إعداد التقرير النهائي...",
]

# ألوان
C_BG = "#05090f"
C_PANEL = "#0a141c"
C_BORDER = "#157f86"
C_ACCENT = "#1de9d6"
C_CYAN = "#39c2d3"
C_DIM = "#7fa6ad"
C_TRUTH = "#1de9d6"
C_MID = "#f5c242"
C_LIE = "#ff3d6e"


KEY_GROUPS = {
    "lips": (61, 291, 13, 14, 0, 17, 78, 308, 82, 312),
    "eyes": (33, 133, 159, 145, 153, 160, 158, 144, 362, 263, 386, 374, 380, 385, 387, 373),
    "nose": (1, 4, 6, 168, 2),
    "brows": (105, 66, 334, 296),
}
KEY_IDX = [i for g in KEY_GROUPS.values() for i in g]
_FEATURE_W = [1.5] * len(KEY_GROUPS["lips"]) + [1.5] * len(KEY_GROUPS["eyes"]) + \
             [1.0] * len(KEY_GROUPS["nose"]) + [1.0] * len(KEY_GROUPS["brows"])
HEAD_SHARE = 0.35  # حصة حركة الرأس من الوزن الكلي (وإلا تذوب نقطة واحدة بين عشرات النقاط)
KEY_W = np.array(_FEATURE_W + [sum(_FEATURE_W) * HEAD_SHARE / (1 - HEAD_SHARE)], dtype=np.float32)

REF_SPEED = 0.30  
REF_STD = 0.040    
REF_DISP = 0.060   
MIX = (0.45, 0.35, 0.20)  
FROZEN_STRETCH = 1.2      
PCT_FLOOR = 8
PCT_CEIL = 97



VOICE_SR = 16000
REF_V_PITCH = 2.0    
REF_V_CV = 0.60      
REF_V_PAUSE = 0.20  
V_MIX = (0.45, 0.30, 0.25)   
FACE_W, VOICE_W = 0.6, 0.4  

CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")


def load_calibration():
    """يحمّل القيم المرجعية المحفوظة (إن وجدت). يعيد True عند النجاح."""
    global REF_SPEED, REF_STD, REF_DISP, REF_V_PITCH, REF_V_CV, REF_V_PAUSE
    try:
        with open(CALIB_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        REF_SPEED, REF_STD, REF_DISP = float(d["speed"]), float(d["std"]), float(d["disp"])
        REF_V_PITCH = float(d.get("v_pitch", REF_V_PITCH))
        REF_V_CV = float(d.get("v_cv", REF_V_CV))
        REF_V_PAUSE = float(d.get("v_pause", REF_V_PAUSE))
        return True
    except Exception:
        return False


def save_calibration(speed, std, disp, voice=None):
    global REF_SPEED, REF_STD, REF_DISP, REF_V_PITCH, REF_V_CV, REF_V_PAUSE
    # حدّ أدنى صغير لتجنب القسمة على قيم شبه صفرية
    REF_SPEED, REF_STD, REF_DISP = max(speed, 0.02), max(std, 0.002), max(disp, 0.003)
    if voice and voice.get("ok"):
        if voice["pitch_st"] is not None:
            REF_V_PITCH = max(voice["pitch_st"], 0.3)
        REF_V_CV = max(voice["loud_cv"], 0.10)
        REF_V_PAUSE = max(voice["pause"], 0.05)
    try:
        with open(CALIB_PATH, "w", encoding="utf-8") as f:
            json.dump({"speed": REF_SPEED, "std": REF_STD, "disp": REF_DISP,
                       "v_pitch": REF_V_PITCH, "v_cv": REF_V_CV, "v_pause": REF_V_PAUSE}, f)
    except Exception:
        pass


def estimate_pitch(x, sr=VOICE_SR, fmin=70.0, fmax=400.0):
    """تقدير طبقة الصوت (Hz) لكتلة صوتية بالارتباط الذاتي. يعيد None إن لم يكن صوتاً مجهوراً."""
    x = x.astype(np.float32) - float(np.mean(x))
    if float(np.max(np.abs(x))) < 1e-4:
        return None
    n = len(x)
    f = np.fft.rfft(x * np.hanning(n), 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(sr / fmax), min(int(sr / fmin), n - 2)
    seg = ac[lo:hi]
    if len(seg) < 3:
        return None
    i = int(np.argmax(seg))
    if seg[i] < 0.35:
        return None
    lag = float(lo + i)
    if 0 < i < len(seg) - 1:  # استيفاء قطع مكافئ لدقة أعلى
        y0, y1, y2 = seg[i - 1], seg[i], seg[i + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-9:
            lag += 0.5 * (y0 - y2) / den
    return sr / lag


def analyze_voice(blocks, sr=VOICE_SR):
    """يحلل كتل الصوت المسجّلة أثناء الفحص.

    المؤشرات: تذبذب طبقة الصوت (نصف-تون)، تذبذب الشدة، ونسبة الوقفات داخل الإجابة.
    يعيد None إذا لم يوجد تسجيل، و{"ok": False} إذا لم يُسمع كلام كافٍ.
    """
    if len(blocks) < 8:
        return None
    rms = np.array([math.sqrt(float(np.mean(b * b)) + 1e-12) for b in blocks])
    # عتبة الكلام: أعلى من ضجيج الخلفية، لكن لا تتجاوز 40% من أعلى مستوى حتى لو تكلّم الشخص طوال الوقت
    thr = max(0.012, min(float(np.percentile(rms, 10)) * 2.5, 0.4 * float(np.percentile(rms, 95))))
    voiced = rms > thr
    out = {"ok": False, "voiced_frac": float(voiced.mean())}
    if int(voiced.sum()) < 6:
        return out
    idx = np.where(voiced)[0]
    pause = 1.0 - float(voiced[idx[0]:idx[-1] + 1].mean())
    vr = rms[voiced]
    loud_cv = float(vr.std() / (vr.mean() + 1e-9))
    f0 = [p for p in (estimate_pitch(blocks[i], sr) for i in idx) if p]
    pitch_st = None
    if len(f0) >= 5:
        pitch_st = float((12 * np.log2(np.array(f0) / np.median(f0))).std())
    parts = [(V_MIX[1], loud_cv / REF_V_CV), (V_MIX[2], pause / REF_V_PAUSE)]
    if pitch_st is not None:
        parts.append((V_MIX[0], pitch_st / REF_V_PITCH))
    ratio = sum(w * r for w, r in parts) / sum(w for w, _ in parts)
    out.update({"ok": True, "pitch_st": pitch_st, "loud_cv": loud_cv, "pause": pause,
                "ratio": ratio, "loud": float(np.percentile(rms, 90))})
    return out


def tension_percentage(ratio):
    """نسبة النشاط -> مؤشر التوتر. 1.0 (حركة طبيعية) ~ 8%، والابتعاد عنها يرفعها.

    z = log2(ratio): z=+1 يعني حركة ضعف الطبيعي (~64%)، z=+1.5 (~85%)، z=+2 (~95%).
    الجمود (z سالب) يُحسب بنفس الشكل لكن بتمديد FROZEN_STRETCH.
    """
    z = math.log2(max(ratio, 1e-3))
    d = z if z >= 0 else abs(z) / FROZEN_STRETCH
    return PCT_FLOOR + (PCT_CEIL - PCT_FLOOR) * (1.0 - math.exp(-d * d))


class MotionTracker:
    """يتتبّع حركة نقاط الوجه. يعمل دائماً للقراءة الحية، ويسجّل فقط أثناء الفحص."""

    def __init__(self):
        self.recording = False
        self._reset_live()
        self._reset_record()

    def _reset_live(self):
        self._prev = None
        self._prev_t = None
        self._origin = None
        self._ipd = None

    def _reset_record(self):
        self.baseline = None
        self.series = []
        self.disp = []
        self.steps = []
        self.frames = 0
        self.face_frames = 0

    # -- التحكم --
    def start_recording(self):
        """يبدأ التسجيل: أول إطار بعدها يصبح الـ Baseline."""
        self._reset_record()
        self._prev = None
        self._origin = None
        self.recording = True

    def lost(self):
        if self.recording:
            self.frames += 1
        self._prev = None

    # -- التطبيع --
    def _normalize(self, pts):
        eye_vec = pts[263] - pts[33]
        ipd = float(np.linalg.norm(eye_vec))
        if ipd < 1e-6:
            return None
       
        self._ipd = ipd if self._ipd is None else 0.9 * self._ipd + 0.1 * ipd
        ipd = self._ipd
        roll = (pts[362] + pts[263]) / 2.0 - (pts[33] + pts[133]) / 2.0
        ang = math.atan2(float(roll[1]), float(roll[0]))
        c, s = math.cos(-ang), math.sin(-ang)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        nose = pts[1]
        if self._origin is None:
            self._origin = nose.copy()
        feat = ((pts[KEY_IDX] - nose) @ rot.T) / ipd           
        head = ((nose - self._origin) / ipd)[None, :]           
        return np.concatenate([feat, head], axis=0).astype(np.float32)

    def update(self, pts, t):
        """pts: مصفوفة Nx2 بالبكسل أو None. يعيد السرعة اللحظية (للعرض) أو None."""
        if pts is None:
            self.lost()
            return None
        norm = self._normalize(pts)
        return self._push(norm, KEY_W, t)

    def update_box(self, center, size, t):
        """وضع احتياطي (Haar): نقطة واحدة تمثل مركز الوجه."""
        if self._origin is None:
            self._origin = np.array(center, dtype=np.float32)
            self._ipd = float(size)
        norm = ((np.array(center, dtype=np.float32) - self._origin) / (self._ipd / 2.0))[None, :]
        return self._push(norm, np.ones(1, dtype=np.float32), t)

    def _push(self, norm, w, t):
        if norm is None:
            self.lost()
            return None
        speed = None
        if self._prev is not None and self._prev.shape == norm.shape:
            dt = t - self._prev_t
            if 0.0 < dt <= 0.25:
                diff = np.abs(norm - self._prev).mean(axis=1)          
                speed = float(np.average(diff, weights=w) / dt)
        if self.recording:
            self.frames += 1
            self.face_frames += 1
            if self.baseline is None:
                self.baseline = norm.copy()                             
            self.series.append(norm)
            self.disp.append(float(np.average(np.abs(norm - self.baseline).mean(axis=1), weights=w)))
            if speed is not None:
                self.steps.append(speed)
        self._prev = norm
        self._prev_t = t
        self._w = w
        return speed

    def finish(self):
        """ينهي التسجيل ويعيد dict بالنتيجة، أو None إذا لم تكفِ البيانات."""
        self.recording = False
        if len(self.series) < 10 or len(self.steps) < 5:
            return None
        arr = np.stack(self.series)                                      
        w = self._w
        steps = np.asarray(self.steps, dtype=np.float32)
        speed = float(0.5 * steps.mean() + 0.5 * np.percentile(steps, 90))
        spread = float(np.average(arr.std(axis=0).mean(axis=1), weights=w))   
        drift = float(np.mean(self.disp))
        ratio = MIX[0] * speed / REF_SPEED + MIX[1] * spread / REF_STD + MIX[2] * drift / REF_DISP
        return {"pct": tension_percentage(ratio), "ratio": ratio, "speed": speed,
                "std": spread, "disp": drift,
                "coverage": self.face_frames / max(1, self.frames)}


def final_score(round_scores):
    """النتيجة النهائية: نصف المتوسط + نصف أعلى جولة (حتى لا تُخفي الجولات الهادئة جولة متوترة)."""
    if not round_scores:
        return 0
    return int(round(0.5 * (sum(round_scores) / len(round_scores)) + 0.5 * max(round_scores)))


def verdict_for(pct):
    if pct < 20:
        return "أعصابك من حديد", "هدوء تام في الوجه والصوت. ولا شعرة تحركت.", C_TRUTH, "truth"
    if pct < 40:
        return "رايق ومرتاح", "حركة ونبرة ضمن المعدل الطبيعي.", C_TRUTH, "truth"
    if pct < 60:
        return "في شوية توتر", "مؤشرات متذبذبة بين الهدوء والتوتر.", C_MID, "mid"
    if pct < 80:
        return "قلبك عم يدق", "رصدنا حركة ونبرة متقلبة في أكثر من جولة.", C_LIE, "lie"
    return "أعصابك ولّعت", "حركة ونبرة صوت عند أقصى حد. خذ نفس عميق!", C_LIE, "lie"



SR = 44100


def _t(dur):
    return np.linspace(0, dur, int(SR * dur), endpoint=False)


def tone(freq, dur, vol=0.4, decay=4.0):
    t = _t(dur)
    w = np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * freq * 2 * t)
    env = np.exp(-decay * t / dur) * np.minimum(1.0, t / 0.005)
    return (w * env * vol).astype(np.float32)


def sweep(f0, f1, dur, vol=0.4):
    t = _t(dur)
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * dur))
    env = np.minimum(1.0, t / 0.01) * np.minimum(1.0, (dur - t) / 0.08)
    return (np.sin(phase) * env * vol).astype(np.float32)


def seq(*parts):
    return np.concatenate(parts).astype(np.float32)


def mix(*parts):
    n = max(len(p) for p in parts)
    out = np.zeros(n, dtype=np.float32)
    for p in parts:
        out[:len(p)] += p
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / max(1.0, peak / 0.8)).astype(np.float32)


class SoundFX:
    def __init__(self):
        self.enabled = sd is not None
        self.sounds = {}
        if not self.enabled:
            return
        silence = np.zeros(int(SR * 0.04), dtype=np.float32)
        self.sounds = {
            "blip": tone(1200, 0.05, 0.22, 6),
            "tick": tone(700, 0.07, 0.25, 5),
            "scan_start": sweep(400, 1300, 0.35, 0.3),
            "success": seq(tone(659, 0.10, 0.35), silence, tone(784, 0.10, 0.35), silence,
                           tone(988, 0.10, 0.35), silence, tone(1319, 0.30, 0.35, 5)),
            "error": seq(tone(220, 0.15, 0.4, 2), silence, tone(196, 0.25, 0.4, 2)),
            "truth": mix(tone(523, 1.2, 0.35, 3), tone(659, 1.2, 0.3, 3),
                         tone(784, 1.2, 0.3, 3), tone(1046, 1.2, 0.2, 3)),
            "mid": seq(tone(440, 0.25, 0.4, 3), tone(554, 0.5, 0.4, 3)),
            "lie": mix(sweep(900, 160, 0.9, 0.4), tone(70, 1.0, 0.5, 3)),
        }

    def play(self, name):
        if not self.enabled or name not in self.sounds:
            return
        try:
            sd.play(self.sounds[name], SR, blocking=False)
        except Exception:
            pass



class MirrorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(800, 520)

        self.cap = None
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_camera)
        self.audio_stream = None
        self.audio_level = 0.0
        self.audio_available = False
        self.audio_blocks = []
        self.audio_recording = False
        self.last_block = None
        self.live_pitch = None
        self._pitch_tick = 0
        self.voice_missing = False

        self.sfx = SoundFX()
        self.deck = QuestionDeck(load_questions())
        self.tracker = MotionTracker()
        self.face_mesh = None
        self.contour_idx = []
        self._face_cascade = None
        self._init_face_tracker()

        
        self.state = "idle"   
        self.questions = []
        self.round_idx = 0
        self.round_scores = []
        self.final_pct = 0
        self.scan_audio = []
        self.scan_t0 = 0.0
        self._last_remain = -1
        self.live_motion = 0.0
        self.debug = False
        self.last_result = None
        self.calibrating = False
        self.calibrated = load_calibration()

        self.scan_timer = QTimer(self)
        self.scan_timer.timeout.connect(self.scan_tick)
        self.analysis_timer = QTimer(self)
        self.analysis_timer.timeout.connect(self.analysis_tick)
        self.count_timer = QTimer(self)
        self.count_timer.timeout.connect(self.count_tick)
        self.auto_timer = QTimer(self)
        self.auto_timer.timeout.connect(self.auto_tick)

        self.build_ui()
        self.start_camera()
        self.start_audio()
        self.show_idle()

    def build_ui(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background: {C_BG}; color: #e8f7f5; }}
            QLabel {{ color: #e8f7f5; background: transparent; }}
            QFrame#panel {{ background: {C_PANEL}; border: 1px solid {C_BORDER}; border-radius: 6px; }}
            QLabel#title {{ color: {C_ACCENT}; font-size: 28px; font-weight: 800; }}
            QLabel#subtitle {{ color: {C_DIM}; font-size: 12px; }}
            QLabel#metric {{ color: {C_CYAN}; font-size: 22px; font-weight: 700; }}
            QLabel#section {{ color: {C_ACCENT}; font-size: 14px; font-weight: 700; }}
            QPushButton {{ background: #0e5f66; color: #ffffff; border: 1px solid {C_ACCENT};
                           border-radius: 4px; padding: 14px 18px; font-size: 18px; font-weight: 700; }}
            QPushButton:hover {{ background: #138a92; }}
            QPushButton:disabled {{ background: #1a2a2d; color: #5f7f83; border-color: #2a4a4d; }}
            QProgressBar {{ background: #0b1d24; border: 1px solid {C_BORDER}; border-radius: 3px;
                            height: 12px; text-align: center; color: transparent; }}
            QProgressBar::chunk {{ background: {C_ACCENT}; }}
        """)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.stack.addWidget(self.build_stage_page())
        self.stack.addWidget(self.build_analysis_page())
        self.stack.addWidget(self.build_result_page())

    def build_stage_page(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(8)

        header = QHBoxLayout()
        logo_label = QLabel()
        logo_label.setFixedSize(60, 60)
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")
        if os.path.exists(logo_path):
            logo_label.setPixmap(QPixmap(logo_path).scaled(56, 56, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            logo_label.setAlignment(Qt.AlignCenter)
        header.addWidget(logo_label, alignment=Qt.AlignVCenter)
        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        title = QLabel(APP_TITLE)
        title.setObjectName("title")
        subtitle = QLabel("THE AI LIE DETECTOR MIRROR  |  تجربة تفاعلية ترفيهية")
        subtitle.setObjectName("subtitle")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        header.addLayout(title_col)
        header.addStretch()
        self.status = QLabel("الحالة: جاهز")
        self.status.setStyleSheet(f"color:{C_ACCENT}; font-weight:700; font-size:14px;")
        header.addWidget(self.status, alignment=Qt.AlignVCenter)
        outer.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(10)
        outer.addLayout(body, 1)

        # الكاميرا
        left = QFrame()
        left.setObjectName("panel")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.setSpacing(6)
        cam_title = QLabel("البث المباشر")
        cam_title.setObjectName("section")
        ll.addWidget(cam_title)
        self.camera_view = QLabel("جارٍ تشغيل الكاميرا...")
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.camera_view.setMinimumSize(320, 240)
        self.camera_view.setStyleSheet(f"background:#02070a; border:1px solid {C_BORDER}; border-radius:4px;")
        ll.addWidget(self.camera_view, 1)
        self.camera_hint = QLabel("الوجه: بانتظار اكتشافه")
        self.camera_hint.setStyleSheet(f"color:{C_DIM};")
        ll.addWidget(self.camera_hint)
        body.addWidget(left, 7)

        # اللوحة الجانبية
        right = QVBoxLayout()
        right.setSpacing(10)
        body.addLayout(right, 4)

        qp = QFrame()
        qp.setObjectName("panel")
        ql = QVBoxLayout(qp)
        ql.setContentsMargins(16, 14, 16, 14)
        ql.setSpacing(8)

        round_row = QHBoxLayout()
        self.round_label = QLabel("ROUND 0 / %d" % ROUNDS)
        self.round_label.setObjectName("section")
        round_row.addWidget(self.round_label)
        round_row.addStretch()
        pips = QHBoxLayout()
        pips.setSpacing(5)
        self.pips = []
        for _ in range(ROUNDS):
            p = QLabel()
            p.setFixedSize(26, 8)
            pips.addWidget(p)
            self.pips.append(p)
        round_row.addLayout(pips)
        ql.addLayout(round_row)

        self.question_label = QLabel("")
        self.question_label.setWordWrap(True)
        self.question_label.setAlignment(Qt.AlignCenter)
        self.question_label.setStyleSheet("font-size:26px; font-weight:700; min-height:110px;")
        ql.addWidget(self.question_label, 1)

        self.timer_label = QLabel(str(SCAN_SECONDS))
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet(f"color:{C_CYAN}; font-size:64px; font-weight:800;")
        ql.addWidget(self.timer_label)
        self.scan_bar = QProgressBar()
        self.scan_bar.setRange(0, 1000)
        self.scan_bar.setValue(0)
        ql.addWidget(self.scan_bar)
        self.hint_label = QLabel("")
        self.hint_label.setWordWrap(True)
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setStyleSheet(f"color:{C_MID}; font-size:13px; min-height:18px;")
        ql.addWidget(self.hint_label)
        self.start_btn = QPushButton("ابدأ التجربة")
        self.start_btn.clicked.connect(self.primary_action)
        ql.addWidget(self.start_btn)
        right.addWidget(qp, 5)

        mp_ = QFrame()
        mp_.setObjectName("panel")
        ml = QVBoxLayout(mp_)
        ml.setContentsMargins(16, 12, 16, 12)
        mt = QLabel("مؤشرات لحظية")
        mt.setObjectName("section")
        ml.addWidget(mt)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        grid.addWidget(QLabel("حركة الوجه"), 0, 0)
        self.motion_label = QLabel("—")
        self.motion_label.setObjectName("metric")
        grid.addWidget(self.motion_label, 0, 1)
        grid.addWidget(QLabel("مستوى الصوت"), 1, 0)
        self.audio_label = QLabel("—")
        self.audio_label.setObjectName("metric")
        grid.addWidget(self.audio_label, 1, 1)
        grid.addWidget(QLabel("طبقة الصوت"), 2, 0)
        self.pitch_label = QLabel("—")
        self.pitch_label.setObjectName("metric")
        grid.addWidget(self.pitch_label, 2, 1)
        ml.addLayout(grid)
        self.audio_bar = QProgressBar()
        self.audio_bar.setRange(0, 100)
        self.audio_bar.setValue(0)
        ml.addWidget(self.audio_bar)
        self.debug_label = QLabel("")
        self.debug_label.setStyleSheet(f"color:{C_DIM}; font-family:Consolas; font-size:11px;")
        self.debug_label.setWordWrap(True)
        self.debug_label.hide()
        ml.addWidget(self.debug_label)
        right.addWidget(mp_, 2)

        footer = QLabel("الخصوصية: المعالجة محلية على جهازك. لا يتم حفظ الفيديو أو الصوت.   |   F11: ملء الشاشة   |   Esc: خروج من ملء الشاشة   |   C: معايرة")
        footer.setStyleSheet("color:#4d6f78; font-size:11px;")
        outer.addWidget(footer)
        return page

    def build_analysis_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(40, 20, 40, 20)
        lay.addStretch(2)
        self.an_tag = QLabel("DATA ANALYSIS")
        self.an_tag.setAlignment(Qt.AlignCenter)
        self.an_tag.setStyleSheet(f"color:{C_CYAN}; font-size:18px; font-weight:700; letter-spacing:4px;")
        lay.addWidget(self.an_tag)
        self.an_title = QLabel("تحليل البيانات")
        self.an_title.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.an_title)
        lay.addSpacing(18)
        self.an_msg = QLabel("")
        self.an_msg.setAlignment(Qt.AlignCenter)
        self.an_msg.setWordWrap(True)
        lay.addWidget(self.an_msg)
        lay.addSpacing(24)
        self.an_bar = QProgressBar()
        self.an_bar.setRange(0, 1000)
        self.an_bar.setFixedHeight(16)
        bar_row = QHBoxLayout()
        bar_row.addStretch(1)
        bar_row.addWidget(self.an_bar, 3)
        bar_row.addStretch(1)
        lay.addLayout(bar_row)
        lay.addSpacing(14)
        self.an_hex = QLabel("")
        self.an_hex.setAlignment(Qt.AlignCenter)
        self.an_hex.setStyleSheet(f"color:{C_DIM}; font-family:Consolas; font-size:15px;")
        lay.addWidget(self.an_hex)
        lay.addStretch(3)
        return page

    def build_result_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(40, 20, 40, 16)
        lay.addStretch(1)
        self.rs_caption = QLabel("مؤشر التوتر")
        self.rs_caption.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.rs_caption)
        self.rs_pct = QLabel("0%")
        self.rs_pct.setAlignment(Qt.AlignCenter)
        self.rs_glow = QGraphicsDropShadowEffect(self.rs_pct)
        self.rs_glow.setOffset(0, 0)
        self.rs_glow.setBlurRadius(90)
        self.rs_pct.setGraphicsEffect(self.rs_glow)
        lay.addWidget(self.rs_pct)
        self.rs_verdict = QLabel("")
        self.rs_verdict.setAlignment(Qt.AlignCenter)
        self.rs_verdict.setWordWrap(True)
        lay.addWidget(self.rs_verdict)
        self.rs_detail = QLabel("")
        self.rs_detail.setAlignment(Qt.AlignCenter)
        self.rs_detail.setWordWrap(True)
        lay.addWidget(self.rs_detail)
        lay.addSpacing(18)
        self.rs_rounds = QLabel("")
        self.rs_rounds.setAlignment(Qt.AlignCenter)
        self.rs_rounds.setTextFormat(Qt.RichText)
        lay.addWidget(self.rs_rounds)
        lay.addStretch(1)
        self.rs_note = QLabel("مؤشر ترفيهي لمستوى التوتر (حركة الوجه + نبرة الصوت). لا يكشف الكذب ولا الصدق.")
        self.rs_note.setAlignment(Qt.AlignCenter)
        self.rs_note.setStyleSheet(f"color:{C_DIM}; font-size:13px;")
        lay.addWidget(self.rs_note)
        self.rs_btn = QPushButton("تجربة جديدة")
        self.rs_btn.clicked.connect(self.show_idle)
        row = QHBoxLayout()
        row.addStretch(2)
        row.addWidget(self.rs_btn, 1)
        row.addStretch(2)
        lay.addLayout(row)
        self.rs_auto = QLabel("")
        self.rs_auto.setAlignment(Qt.AlignCenter)
        self.rs_auto.setStyleSheet("color:#4d6f78; font-size:12px;")
        lay.addWidget(self.rs_auto)
        self.result_color = C_ACCENT
        return page

    def apply_scaling(self):
        h = max(520, self.height())
        self.an_title.setStyleSheet(f"color:{C_ACCENT}; font-size:{int(h*0.09)}px; font-weight:800;")
        self.an_msg.setStyleSheet(f"color:#e8f7f5; font-size:{int(h*0.036)}px; font-weight:600;")
        self.rs_caption.setStyleSheet(f"color:{C_DIM}; font-size:{int(h*0.04)}px; font-weight:700; letter-spacing:6px;")
        self.rs_pct.setStyleSheet(f"color:{self.result_color}; font-size:{int(h*0.30)}px; font-weight:900;")
        self.rs_verdict.setStyleSheet(f"color:{self.result_color}; font-size:{int(h*0.085)}px; font-weight:800;")
        self.rs_detail.setStyleSheet(f"color:#cfe8e6; font-size:{int(h*0.03)}px;")
        self.rs_rounds.setStyleSheet(f"font-size:{int(h*0.026)}px;")
        self.rs_btn.setStyleSheet(f"font-size:{int(h*0.026)}px; padding:12px 18px;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.apply_scaling()

    # ------------------------------------------------------- Camera / Audio
    def _init_face_tracker(self):
        if mp is not None:
            try:
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
                conns = mp.solutions.face_mesh.FACEMESH_CONTOURS
                self.contour_idx = sorted({i for pair in conns for i in pair})
            except Exception:
                self.face_mesh = None
        if self.face_mesh is None:
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def start_camera(self):
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.camera_timer.start(20)
            else:
                self.camera_view.setText("تعذّر فتح الكاميرا.\nتحقق من توصيلها وصلاحيات Windows.")
                self.status.setText("الحالة: الكاميرا غير متاحة")
                self.status.setStyleSheet(f"color:{C_MID}; font-weight:700;")
        except Exception as e:
            self.camera_view.setText(f"تعذّر تشغيل الكاميرا:\n{e}")

    def start_audio(self):
        if sd is None:
            self.audio_available = False
            self.audio_label.setText("غير متاح")
            return
        try:
            self.audio_stream = sd.InputStream(
                channels=1, samplerate=16000, blocksize=1024, callback=self.audio_callback)
            self.audio_stream.start()
            self.audio_available = True
        except Exception:
            self.audio_available = False
            self.audio_label.setText("غير متاح")

    def audio_callback(self, indata, frames, time_info, status):
        try:
            x = indata[:, 0].astype(np.float32)
            rms = float(np.sqrt(np.mean(x * x) + 1e-12))
            self.audio_level = min(1.0, rms * 8.0)
            self.last_block = x.copy()
            if self.audio_recording:
                self.audio_blocks.append(x.copy())
        except Exception:
            pass

    @staticmethod
    def _brackets(frame, x0, y0, x1, y1, color, thick=3):
        d = max(14, int(min(x1 - x0, y1 - y0) * 0.18))
        for (x, y, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(frame, (x, y), (x + sx * d, y), color, thick)
            cv2.line(frame, (x, y), (x, y + sy * d), color, thick)

    def update_camera(self):
        if self.cap is None or not self.cap.isOpened():
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        cyan = (211, 194, 57)  # BGR
        speed = None
        face_found = False

        if self.face_mesh is not None:
            small = frame if w <= 640 else cv2.resize(frame, (640, int(h * 640 / w)))
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb_small.flags.writeable = False
            res = self.face_mesh.process(rgb_small)
            pts = None
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                norm = np.array([(p.x, p.y) for p in lm], dtype=np.float32)
                pts = norm * np.array([w, h], dtype=np.float32)
            speed = self.tracker.update(pts, now)
            if pts is not None:
                face_found = True
                for i in self.contour_idx:
                    cv2.circle(frame, (int(pts[i][0]), int(pts[i][1])), 1, cyan, -1)
                x0, y0 = pts[:468].min(axis=0).astype(int)
                x1, y1 = pts[:468].max(axis=0).astype(int)
                self._brackets(frame, x0, y0, x1, y1, cyan)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(70, 70))
            if len(faces) > 0:
                x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                face_found = True
                speed = self.tracker.update_box((x + fw / 2, y + fh / 2), fw, now)
                self._brackets(frame, x, y, x + fw, y + fh, cyan)
            else:
                self.tracker.lost()

        # مسح ضوئي أثناء الفحص
        if self.state == "scan":
            prog = min(1.0, (now - self.scan_t0) / SCAN_SECONDS)
            sweep_pos = prog * 4 % 2
            sweep_pos = sweep_pos if sweep_pos <= 1 else 2 - sweep_pos
            y = int(sweep_pos * (h - 1))
            cv2.line(frame, (0, y), (w, y), cyan, 3)
            cv2.line(frame, (0, max(0, y - 14)), (w, max(0, y - 14)), (120, 100, 30), 1)

        if speed is not None:
            self.live_motion = 0.9 * self.live_motion + 0.1 * min(1.0, speed / (2.0 * REF_SPEED))
        self.camera_hint.setText("الوجه: تم اكتشاف وجه أمام الكاميرا" if face_found else "الوجه: بانتظار اكتشافه")
        self.motion_label.setText(f"{int(self.live_motion * 100)}%")
        if self.audio_available:
            self.audio_label.setText(f"{int(self.audio_level * 100)}%")
            self.audio_bar.setValue(int(self.audio_level * 100))
        self._pitch_tick += 1
        if self._pitch_tick % 5 == 0 and self.last_block is not None:
            p = estimate_pitch(self.last_block) if self.audio_level > 0.05 else None
            if p:
                self.live_pitch = p if self.live_pitch is None else 0.6 * self.live_pitch + 0.4 * p
                self.pitch_label.setText(f"{int(self.live_pitch)} Hz")
            elif self.audio_level <= 0.05:
                self.live_pitch = None
                self.pitch_label.setText("—")
        if self.debug:
            txt = "live speed: " + ("-" if speed is None else f"{speed:.2f}") + f"   (REF_SPEED={REF_SPEED})"
            r = self.last_result
            if r:
                txt += (f"\nlast scan: speed {r['speed']:.2f}  std {r['std']:.3f}  disp {r['disp']:.3f}"
                        f"  ratio {r['ratio']:.2f}  ->  {r['pct']:.0f}%")
                v = r.get("voice")
                if v and v.get("ok"):
                    ps = "-" if v["pitch_st"] is None else f"{v['pitch_st']:.2f}"
                    txt += (f"\nvoice: pitch_st {ps}  loud_cv {v['loud_cv']:.2f}  pause {v['pause']:.2f}"
                            f"  ratio {v['ratio']:.2f}")
                elif v is not None:
                    txt += "\nvoice: no speech detected"
            self.debug_label.setText(txt)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(image).scaled(self.camera_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_view.setPixmap(pix)

   
    def set_status(self, text, color=C_ACCENT):
        self.status.setText(f"الحالة: {text}")
        self.status.setStyleSheet(f"color:{color}; font-weight:700; font-size:14px;")

    def update_pips(self):
        for i, p in enumerate(self.pips):
            done = i < len(self.round_scores)
            current = (i == self.round_idx) and self.state in ("ready", "scan")
            if done:
                p.setStyleSheet(f"background:{C_ACCENT}; border:1px solid {C_ACCENT};")
            elif current:
                p.setStyleSheet(f"background:transparent; border:1px solid {C_ACCENT};")
            else:
                p.setStyleSheet("background:#12262b; border:1px solid #1d3d43;")

    def show_idle(self):
        self.auto_timer.stop()
        self.count_timer.stop()
        self.scan_timer.stop()
        self.analysis_timer.stop()
        self.state = "idle"
        self.questions = []
        self.round_idx = 0
        self.round_scores = []
        self.round_label.setText(f"ROUND 0 / {ROUNDS}")
        self.question_label.setText(f"ستجيب على {ROUNDS} أسئلة. في كل جولة اقرأ السؤال واضغط «ابدأ الفحص» ثم أجب أمام الكاميرا.")
        self.question_label.setStyleSheet("font-size:22px; font-weight:600; min-height:110px; color:#cfe8e6;")
        self.timer_label.setText(str(SCAN_SECONDS))
        self.scan_bar.setValue(0)
        self.hint_label.setText("" if self.calibrated else "لم تتم المعايرة بعد: قف أمام الكاميرا واضغط C وتحدث بشكل طبيعي.")
        self.start_btn.setText("ابدأ التجربة")
        self.start_btn.setEnabled(True)
        self.set_status("جاهز")
        self.update_pips()
        self.stack.setCurrentIndex(0)

    def primary_action(self):
        if self.state == "idle":
            self.begin_session()
        elif self.state == "ready":
            self.start_scan()

    def begin_session(self):
        self.questions = self.deck.draw(ROUNDS)
        self.round_idx = 0
        self.round_scores = []
        self.sfx.play("blip")
        self.show_round()

    def show_round(self):
        self.state = "ready"
        self.round_label.setText(f"ROUND {self.round_idx + 1} / {ROUNDS}")
        self.question_label.setText(self.questions[self.round_idx])
        self.question_label.setStyleSheet("font-size:28px; font-weight:700; min-height:110px; color:#ffffff;")
        self.timer_label.setText(str(SCAN_SECONDS))
        self.scan_bar.setValue(0)
        self.start_btn.setText("ابدأ الفحص")
        self.start_btn.setEnabled(True)
        self.hint_label.setText("لم نسمع صوتك في الجولة السابقة. أجب بصوت واضح." if self.voice_missing and self.round_idx > 0 else "")
        self.set_status(f"الجولة {self.round_idx + 1} جاهزة")
        self.update_pips()
        self.stack.setCurrentIndex(0)

    def start_scan(self, calibrate=False):
        now = time.time()
        self.calibrating = calibrate
        self._state_before = self.state
        self.tracker.start_recording()   # أول إطار وجه بعد هذه اللحظة = Baseline
        self.scan_audio = []
        self.audio_blocks = []
        self.audio_recording = False
        # نبدأ تسجيل الصوت بعد انتهاء صوت بداية الفحص حتى لا يلتقطه الميكروفون
        QTimer.singleShot(450, self._begin_voice_recording)
        self.hint_label.setText("")
        self.state = "scan"
        self.scan_t0 = now
        self._last_remain = -1
        self.start_btn.setEnabled(False)
        self.start_btn.setText("جاري المعايرة..." if calibrate else "جاري الفحص...")
        self.set_status("معايرة الحركة الطبيعية" if calibrate else "فحص الكاميرا قيد التشغيل", C_CYAN)
        self.update_pips()
        self.sfx.play("scan_start")
        self.scan_timer.start(40)

    def _begin_voice_recording(self):
        if self.state == "scan":
            self.audio_recording = True

    def scan_tick(self):
        el = time.time() - self.scan_t0
        self.scan_bar.setValue(int(min(1.0, el / SCAN_SECONDS) * 1000))
        remain = max(0, math.ceil(SCAN_SECONDS - el))
        if remain != self._last_remain:
            self._last_remain = remain
            self.timer_label.setText(str(remain))
        self.scan_audio.append(self.audio_level)
        if el >= SCAN_SECONDS:
            self.scan_timer.stop()
            self.finish_scan()

    def finish_scan(self):
        result = self.tracker.finish()
        self.audio_recording = False
        voice = analyze_voice(self.audio_blocks) if self.audio_available else None
        if self.calibrating:
            self.calibrating = False
            self.state = self._state_before
            self.start_btn.setEnabled(True)
            self.start_btn.setText("ابدأ التجربة" if self.state == "idle" else "ابدأ الفحص")
            self.timer_label.setText(str(SCAN_SECONDS))
            self.scan_bar.setValue(0)
            if result is None or result["coverage"] < MIN_FACE_COVERAGE:
                self.hint_label.setText("فشلت المعايرة: لم يتم رصد الوجه. أعد الضغط على C.")
                self.sfx.play("error")
            else:
                save_calibration(result["speed"], result["std"], result["disp"], voice)
                self.calibrated = True
                self.hint_label.setText("تمت المعايرة وحُفظت.")
                self.sfx.play("success")
            self.set_status("جاهز")
            self.update_pips()
            return
        if result is None or result["coverage"] < MIN_FACE_COVERAGE:
            self.state = "ready"
            self.hint_label.setText("لم يتم رصد وجهك بوضوح. قف أمام الكاميرا بإضاءة جيدة وأعد الفحص.")
            self.start_btn.setText("إعادة الفحص")
            self.start_btn.setEnabled(True)
            self.timer_label.setText(str(SCAN_SECONDS))
            self.scan_bar.setValue(0)
            self.set_status("لم يتم رصد الوجه", C_MID)
            self.sfx.play("error")
            self.update_pips()
            return
        ratio = result["ratio"]
        has_voice = bool(voice and voice.get("ok"))
        if has_voice:
            ratio = FACE_W * ratio + VOICE_W * voice["ratio"]
        result["pct"] = tension_percentage(ratio)
        result["voice"] = voice
        self.voice_missing = self.audio_available and not has_voice
        self.last_result = result
        self.round_scores.append(result["pct"])
        self.state = "busy"
        self.sfx.play("success")
        self.update_pips()
        self.start_analysis(final=len(self.round_scores) >= ROUNDS)

    def start_analysis(self, final):
        self.an_final = final
        self.an_dur = FINAL_ANALYSIS_SECONDS if final else ROUND_ANALYSIS_SECONDS
        if final:
            self.an_msgs = list(ANALYSIS_MESSAGES)
            self.an_tag.setText("FINAL ANALYSIS")
        else:
            self.an_msgs = random.sample(ANALYSIS_MESSAGES, 3)
            self.an_tag.setText(f"DATA ANALYSIS  |  ROUND {len(self.round_scores)} / {ROUNDS}")
        self.an_t0 = time.time()
        self._an_idx = -1
        self.an_bar.setValue(0)
        self.set_status("جاري التحليل", C_CYAN)
        self.stack.setCurrentIndex(1)
        self.analysis_timer.start(60)

    def analysis_tick(self):
        el = time.time() - self.an_t0
        prog = min(1.0, el / self.an_dur)
        self.an_bar.setValue(int(prog * 1000))
        idx = min(len(self.an_msgs) - 1, int(prog * len(self.an_msgs)))
        if idx != self._an_idx:
            self._an_idx = idx
            self.an_msg.setText(self.an_msgs[idx])
            self.sfx.play("blip")
        self.an_hex.setText("  ".join(f"0x{random.randint(0, 0xFFFF):04X}" for _ in range(6)))
        if el >= self.an_dur:
            self.analysis_timer.stop()
            if self.an_final:
                self.show_result()
            else:
                self.round_idx += 1
                self.show_round()

    def show_result(self):
        self.state = "result"
        self.final_pct = final_score(self.round_scores)
        headline, detail, color, kind = verdict_for(self.final_pct)
        self.result_color = color
        self.rs_glow.setColor(QColor(color))
        self.rs_verdict.setText(headline)
        self.rs_detail.setText(detail)
        cells = []
        for i, s in enumerate(self.round_scores):
            c = verdict_for(int(s))[2]
            cells.append(f"<span style='color:{C_DIM}'>R{i + 1}</span> "
                         f"<span style='color:{c}; font-weight:700'>{int(round(s))}%</span>")
        self.rs_rounds.setText("&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;".join(cells))
        self.rs_pct.setText("0%")
        self.apply_scaling()
        self.set_status("النتيجة النهائية", color)
        self.stack.setCurrentIndex(2)
        self.sfx.play(kind)
        self.count_t0 = time.time()
        self.count_timer.start(16)
        self.auto_left = AUTO_RESET_SECONDS
        self.rs_auto.setText(f"العودة التلقائية بعد {self.auto_left} ثانية")
        self.auto_timer.start(1000)

    def count_tick(self):
        p = min(1.0, (time.time() - self.count_t0) / 1.6)
        eased = 1 - (1 - p) ** 3
        self.rs_pct.setText(f"{int(round(self.final_pct * eased))}%")
        if p >= 1.0:
            self.count_timer.stop()

    def auto_tick(self):
        self.auto_left -= 1
        if self.auto_left <= 0:
            self.auto_timer.stop()
            self.show_idle()
        else:
            self.rs_auto.setText(f"العودة التلقائية بعد {self.auto_left} ثانية")

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_F11:
            self.toggle_fullscreen()
        elif k == Qt.Key_Escape and self.isFullScreen():
            self.showNormal()
        elif k == Qt.Key_C:
            if self.state in ("idle", "ready") and self.stack.currentIndex() == 0:
                self.start_scan(calibrate=True)
        elif k == Qt.Key_D:
            self.debug = not self.debug
            self.debug_label.setVisible(self.debug)
        elif k in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            if self.stack.currentIndex() == 2:
                self.show_idle()
            else:
                self.primary_action()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        for t in (self.camera_timer, self.scan_timer, self.analysis_timer, self.count_timer, self.auto_timer):
            t.stop()
        if self.cap is not None:
            self.cap.release()
        if self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except Exception:
                pass
        if self.audio_stream is not None:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception:
                pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setLayoutDirection(Qt.RightToLeft)
    window = MirrorWindow()
    window.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
