# Dungeon Studio Project Scope

## Overview
Dungeon Studio is a Django-based platform for automatic music genre classification using an SVM (Support Vector Machine) AI model.

## User Roles
1. **Studio User**: Access to Dashboard, Classification, History, Reports.
2. **Studio Manager / Admin**: Adds Model Performance.
3. **Admin**: Adds User Management.

## Screens to Generate
1. **Dashboard**: Stats summary (Total classifications, top genre, AI model info: 75.5% accuracy, RBF kernel, 43 audio features).
2. **Classification (Core)**: Drag & drop upload (mp3, wav, flac, etc.). Results display predicted genre, confidence %, probability bar chart, and file info.
3. **History**: Table of past predictions with filters and delete actions. Role-based tabs ("My History" vs "All History").
4. **Reports**: Date/genre/user filters. Preview, PDF/Excel export, and history of generated reports.
5. **Model Performance (Detailed)**: 
    - Metric cards (Accuracy 75.5%, Precision 76.81%, Recall 75.5%, F1 75.8%).
    - 5-Fold Cross-Validation bar chart (Avg 74.75%).
    - 10x10 Confusion Matrix Heatmap (GTZAN dataset: Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock).
    - Per-genre Precision/Recall/F1 table with icons.
    - Model info (SVM RBF, 43 features, 800 train / 200 test data).
6. **User Management**: Admin table for user roles and account actions.
7. **Login & Register**: Minimalist authentication forms.

## Visual Identity
- **Background**: #F8FAFC (Off-white)
- **Primary**: #4F46E5 (Indigo)
- **Accent**: #06B6D4 (Cyan)
- **Border Radius**: 16px (2xl)
- **Typography**: Plus Jakarta Sans / Inter
- **Shadows**: Soft, subtle elevation
