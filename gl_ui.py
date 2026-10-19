#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HUD-оверлей превью: скелет рук, рамка критерия, строки статистики."""

import cv2
import mediapipe as mp

from gl_session import KIND_GESTURE, pct

CRITERION_TEXT = "Эталон критерия: обе кисти >= 90% кадров"


def draw_hud(frame, session, pos_label, totals, num_hands, fps,
             landmarks, draw_landmarks: bool) -> None:
    if draw_landmarks and landmarks:
        for hand_lms in landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_lms, mp.solutions.hands.HAND_CONNECTIONS)

    color = (0, 200, 0) if num_hands >= 2 else (0, 120, 255)
    cv2.rectangle(frame, (8, 8), (632, 118), color, 2)

    if session:
        rec_s = int(session["elapsed"])
        line0 = (f"ЗАПИСЬ {rec_s//60:02d}:{rec_s%60:02d} | "
                 f"{'НЕ ЖЕСТ' if session['kind'] != KIND_GESTURE else 'жест'}: "
                 f"{session['gesture']} | {session['position']}")
    else:
        line0 = f"Позиция: {pos_label} | r — жест, n — «не жест», p — положение"
    line1 = (f"Кисти обе: {pct(totals['both'], totals['frames']):5.1f}% | "
             f"Лицо: {pct(totals['face'], totals['frames']):5.1f}% | "
             f"Кадров: {totals['frames']} | ~{fps:4.1f} к/с | рук в кадре: {num_hands}")
    cv2.putText(frame, line0, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(frame, line1, (14, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if not session:
        cv2.putText(frame, CRITERION_TEXT, (14, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
