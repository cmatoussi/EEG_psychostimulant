#!/usr/bin/env python3
import re
import pandas as pd

def _greekify(text: str) -> str:
    rep = {"alpha": "Alpha", "beta": "Beta", "gamma": "Gamma", "theta": "Theta", "delta": "Delta"}
    for k, v in rep.items():
        text = re.sub(rf"\b{k}\b", v, text, flags=re.IGNORECASE)
    return text

def make_label_map_keep_sensor(cols):
    """Build compact labels per column, preserving the sensor name."""
    abbrev = {
        "BandRatiosFromAverageFooof": "(Corrected)",
        "BandRatiosFromAverageSpectrum": "",
        "RelativeBandPowerFromAverageFooof": "(Corrected)",
        "RelativeBandPowerFromAverageSpectrum": "",
        "higuchiFd": "Higuchi FD",
        "katzFd": "Katz FD",
        "petrosianFd": "Petrosian FD",
        "hjorthComplexity": "Hjorth Complexity",
        "hjorthMobility": "Hjorth Mobility",
        "numZerocross": "ZeroCross",
        "svdEntropy": "SVD Entropy",
        "spectralEntropy": "Spectral Entropy",
        "sampleEntropy": "Sample Entropy",
        "permEntropy": "Perm Entropy",
        "entropyMultiscale": "MSE",
        "fooofExponent": "1/f slope",
        "foofOffset": "1/f Offset",
        "lzivComplexity": "LZ Complexity",
    }
    out = {}
    for raw in cols:
        s = raw
        if s.startswith("feature-"):
            s = s[len("feature-") :]
        
        sensor = None
        # Handle trailing '.spaces-<SENSOR>' OR '_ch-<SENSOR>'
        m_sensor = re.search(r"(\.spaces-|_ch-)([-A-Za-z0-9]+)$", s)
        if m_sensor:
            sensor = m_sensor.group(2)
            s = s[: m_sensor.start()]
            
        s = s.replace(".bands-", " ").replace("Epochs", " ")
        m_pair = re.search(r"bands_pairs-\((.+)\)", s)
        if m_pair:
            pair = m_pair.group(1).replace("'", "").replace(" ", "").replace(",", "/")
            pair = _greekify(pair)
            head = s[: m_pair.start()].rstrip(".")
            corrected = False
            for k, v in abbrev.items():
                if k in head:
                    head = head.replace(k, v)
            if "(Corrected)" in head or "(corrected)" in head:
                corrected = True
                head = head.replace("(Corrected)", "").replace("(corrected)", "").strip()
            feat_label = f"{head} {pair}".strip()
            if corrected: feat_label += " (Corrected)"
        else:
            feat_label = s
            corrected = False
            for k, v in abbrev.items():
                if k in feat_label:
                    feat_label = feat_label.replace(k, v)
            if "(Corrected)" in feat_label or "(corrected)" in feat_label:
                corrected = True
                feat_label = feat_label.replace("(Corrected)", "").replace("(corrected)", "").strip()
            feat_label = _greekify(feat_label)
            feat_label = feat_label.replace("MeanEpochs", "")
            feat_label = re.sub(r"[_.-]+$", "", feat_label).strip()
            if corrected: feat_label += " (Corrected)"
            
        feat_label = re.sub(r"\s+", " ", feat_label).strip()
        out[raw] = f"{sensor or ''} — {feat_label}".strip(" —")
    return out

def make_label_map(items):
    out = {}
    full_map = make_label_map_keep_sensor(items)
    for raw, formatted in full_map.items():
        # Remove the leading sensor name 'C3 - ' if it's there
        if " — " in formatted:
            formatted = formatted.split(" — ")[1]
        out[raw] = formatted
    return out

def generate_coords_from_mne(montage="standard_1020", restrict_to=None):
    import mne
    std_montage = mne.channels.make_standard_montage(montage)
    pos = std_montage.get_positions()
    ch_pos = pos.get('ch_pos', {})
    rows = []
    names = list(restrict_to) if restrict_to else list(ch_pos.keys())
    for name in names:
        key = name
        if key not in ch_pos:
            if name.upper() in ch_pos: key = name.upper()
            elif name.capitalize() in ch_pos: key = name.capitalize()
            else: continue
        xyz = ch_pos[key]
        rows.append((name, float(xyz[0]), float(xyz[1])))
    if not rows:
        rows = [(n, float(v[0]), float(v[1])) for n, v in ch_pos.items()]
    return pd.DataFrame(rows, columns=["name", "x", "y"]).set_index("name")
