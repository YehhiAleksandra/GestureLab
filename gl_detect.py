#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Детекция: построение MediaPipe Hands + Face и прогон кадра."""

import mediapipe as mp


def build_detectors(conf: dict):
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=conf["min_detection_confidence"],
        min_tracking_confidence=conf["min_tracking_confidence"],
    )
    face = mp.solutions.face_detection.FaceDetection(
        min_detection_confidence=conf["face_confidence"])
    return hands, face


def landmarks_to_json(multi_hand_landmarks) -> list:
    """[[x0,y0,z0,x1,y1,z1,...x20,y20,z20], ...] — порядок как в Android-версии."""
    out = []
    if not multi_hand_landmarks:
        return out
    for hand in multi_hand_landmarks:
        flat = []
        for lm in hand.landmark:
            flat.extend((lm.x, lm.y, lm.z))
        out.append(flat)
    return out


def process_frame(rgb, hands, face):
    """Возвращает (число кистей, есть ли лицо, ландмарки в формате JSON)."""
    res_h = hands.process(rgb)
    res_f = face.process(rgb)
    return (len(res_h.multi_hand_landmarks or []),
            bool(res_f.detections),
            landmarks_to_json(res_h.multi_hand_landmarks),
            res_h.multi_hand_landmarks)
