@echo off
REM ======================================================================
REM  run_all_personalization.bat
REM  Every personalization experiment, ONE methodology:
REM    - 3 donors per subject (same donors for a subject in every run)
REM    - XGBoost with tuned params from configs\xgb_best_params.json
REM    - first-K night order unless stated
REM  Safe to stop and restart: each run resumes from its _partial.csv.
REM  Results go to results\personalization\, curves to figures\personalization\.
REM  Run from PowerShell (any folder) with:
REM      .\scripts\experiments\run_all_personalization.bat
REM ======================================================================

set RUN=python "%~dp0run_personalization_v2.py"
set WALCH="%~dp0..\..\data\features\walch_features.csv"

echo [1/9] all features
%RUN% --feature-set all --donors 3 --tag final_all

echo [2/9] all features, real-time
%RUN% --feature-set all --realtime --donors 3 --tag final_realtime

echo [3/9] all features, random night order
%RUN% --feature-set all --order random --donors 3 --tag final_random

echo [4/9] Walch features
%RUN% --features %WALCH% --donors 3 --tag final_walch

echo [5/9] personal-reference features only
%RUN% --feature-set personal --donors 3 --tag final_personal

echo [6/9] non-personal features only
%RUN% --feature-set nonpersonal --donors 3 --tag final_nonpersonal

echo [7/9] heart rate only
%RUN% --feature-set hr --donors 3 --tag final_hr

echo [8/9] motion only
%RUN% --feature-set motion --donors 3 --tag final_motion

echo [9/9] heart rate + motion
%RUN% --feature-set hr_motion --donors 3 --tag final_hrmotion

echo.
echo ALL DONE.
