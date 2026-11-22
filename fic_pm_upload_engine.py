import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

SOFTWARE_REQUIRED = ["Software_Name", "Version", "Publisher", "Install_Date",
                     "Install_Location", "Registry_Hive", "Registry_Path"]
USB_REQUIRED = ["Device_Name", "Serial_Number", "Device_Type", "Registry_Path"]
STARTUP_REQUIRED = ["Location", "Value_Name", "Value_Data", "Registry_Path"]
PREFETCH_REQUIRED = ["File_Name", "File_Path", "File_Size", "Last_Modified"]

def validate_frame(df, required, label):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} CSV missing columns: {', '.join(missing)}")
    return df.fillna("").copy()

def run_analysis(software, usb, startup, prefetch):
    software = validate_frame(software, SOFTWARE_REQUIRED, "Software")
    usb = validate_frame(usb, USB_REQUIRED, "USB")
    startup = validate_frame(startup, STARTUP_REQUIRED, "Startup")
    prefetch = validate_frame(prefetch, PREFETCH_REQUIRED, "Prefetch")

    if len(software) < 5:
        raise ValueError("At least 5 software records are required for this prototype's anomaly detectors.")

    # Feature engineering mirrors the local FIC-PM live feature script.
    df = software.copy()
    df["Name_Length"] = df["Software_Name"].astype(str).str.len()
    df["Version_Present"] = df["Version"].astype(str).str.strip().ne("").astype(int)
    df["Publisher_Present"] = df["Publisher"].astype(str).str.strip().ne("").astype(int)
    df["Install_Date_Present"] = df["Install_Date"].astype(str).str.strip().ne("").astype(int)
    df["Install_Location_Present"] = df["Install_Location"].astype(str).str.strip().ne("").astype(int)
    df["Registry_Path_Length"] = df["Registry_Path"].astype(str).str.len()
    df["HKCU_Source"] = df["Registry_Hive"].astype(str).str.upper().str.contains("HKCU").astype(int)
    df["HKLM_Source"] = df["Registry_Hive"].astype(str).str.upper().str.contains("HKLM").astype(int)

    feature_columns = ["Name_Length", "Version_Present", "Publisher_Present",
        "Install_Date_Present", "Install_Location_Present", "Registry_Path_Length",
        "HKCU_Source", "HKLM_Source"]
    X = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    df["IF_Flag"] = (IsolationForest(n_estimators=200, contamination=min(.05, max(1/len(df), .01)),
                                     random_state=42).fit_predict(X_scaled) == -1).astype(int)
    neighbors = min(20, len(df)-1)
    df["LOF_Flag"] = (LocalOutlierFactor(n_neighbors=neighbors,
                            contamination=min(.05, max(1/len(df), .01))).fit_predict(X_scaled) == -1).astype(int)
    df["SVM_Flag"] = (OneClassSVM(kernel="rbf", gamma="scale", nu=min(.05, max(1/len(df), .01)))
                      .fit_predict(X_scaled) == -1).astype(int)
    df["Anomaly_Detector_Count"] = df[["IF_Flag","LOF_Flag","SVM_Flag"]].sum(axis=1).astype(int)
    df["Detector_Consensus"] = df["Anomaly_Detector_Count"]
    df["Detector_Agreement"] = df["Anomaly_Detector_Count"] / 3
    df["Detector_Status"] = df["Anomaly_Detector_Count"].map({
        0:"No detector flagged", 1:"Single detector flag",
        2:"Two detector agreement", 3:"Three detector agreement"})

    def normalize(value):
        return re.sub(r"[^a-z0-9]", "", str(value).lower())

    def extract_exes(value):
        matches = re.findall(r'(?i)([a-z0-9_. -]+\.exe)', str(value))
        return [m.strip().strip('"').split("\\")[-1].lower() for m in matches]

    startup_exes = []
    for _, row in startup.iterrows():
        for exe in extract_exes(row.get("Value_Data", "")):
            startup_exes.append(exe)
    pf_exes = []
    for _, row in prefetch.iterrows():
        name = str(row.get("File_Name", "")).split("-")[0].strip().lower()
        if name.endswith(".pf"): name = name[:-3]
        pf_exes.append((name, str(row.get("File_Name", ""))))

    startup_counts, startup_names, pf_counts, pf_names = [], [], [], []
    for _, row in df.iterrows():
        norm_name = normalize(row["Software_Name"])
        sm = sorted(set(exe for exe in startup_exes
            if len(normalize(exe.replace(".exe",""))) >= 4 and
            (normalize(exe.replace(".exe","")) in norm_name or norm_name in normalize(exe.replace(".exe","")))))
        pm = sorted(set(file for exe,file in pf_exes
            if len(normalize(exe.replace(".exe",""))) >= 4 and
            (normalize(exe.replace(".exe","")) in norm_name or norm_name in normalize(exe.replace(".exe","")))))
        startup_counts.append(len(sm)); startup_names.append("; ".join(sm))
        pf_counts.append(len(pm)); pf_names.append("; ".join(pm))

    df["Startup_Match_Count"] = startup_counts
    df["Startup_Matched_Executables"] = startup_names
    df["Prefetch_Match_Count"] = pf_counts
    df["Prefetch_Matched_Files"] = pf_names
    df["USB_Record_Count"] = len(usb)
    df["Startup_Source_Record_Count"] = len(startup)
    df["Prefetch_Source_Record_Count"] = len(prefetch)
    df["Context_Evidence_Count"] = ((df["Startup_Match_Count"] > 0).astype(int) +
                                    (df["Prefetch_Match_Count"] > 0).astype(int))
    df["Detector_Flag_Count"] = df["Anomaly_Detector_Count"]
    df["Evidence_Review_Indicator"] = ((df["Detector_Flag_Count"] > 0) |
                                       (df["Context_Evidence_Count"] > 0))

    # Experimental 60/20/20 priority score; not a probability or verdict.
    df["Detector_Component"] = (df["Anomaly_Detector_Count"] / 3 * 60).round(2)
    df["Startup_Component"] = (df["Startup_Match_Count"] > 0).astype(int) * 20
    df["Prefetch_Component"] = (df["Prefetch_Match_Count"] > 0).astype(int) * 20
    df["Metadata_Gap_Count"] = sum(
        df[c].astype(str).str.strip().eq("").astype(int)
        for c in ["Version", "Publisher", "Install_Date", "Install_Location"])
    df["FIC_PM_Score"] = (df["Detector_Component"] + df["Startup_Component"] +
                          df["Prefetch_Component"]).round(2)
    df["Priority_Rank"] = df["FIC_PM_Score"].rank(method="min", ascending=False).astype(int)
    df["Priority_Level"] = pd.cut(df["FIC_PM_Score"], bins=[-1,19,39,69,100],
                                  labels=["Routine","Low","Moderate","High"]).astype(str)

    def reasons(row):
        parts=[]
        if row["Anomaly_Detector_Count"] > 0:
            parts.append(f"Detector agreement: {int(row['Anomaly_Detector_Count'])}/3")
        if row["Startup_Match_Count"] > 0: parts.append("Startup name association")
        if row["Prefetch_Match_Count"] > 0: parts.append("Prefetch name association")
        if row["Metadata_Gap_Count"] > 0: parts.append(f"Metadata gaps: {int(row['Metadata_Gap_Count'])}")
        return "; ".join(parts) or "No indicators in current scoring inputs"
    df["Review_Reason"] = df.apply(reasons, axis=1)
    df = df.sort_values(["FIC_PM_Score","Software_Name"], ascending=[False,True]).reset_index(drop=True)
    detector_cols = ["Software_Name","IF_Flag","LOF_Flag","SVM_Flag",
                     "Anomaly_Detector_Count","Detector_Status"]
    detector_results = df[detector_cols].copy()
    return df, detector_results, startup, usb, prefetch
