from django.urls import path
from . import views

app_name = 'Klasifikasi'

urlpatterns = [
    # ── Auth (public) ──────────────────────────────────────────────────────
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_view, name='register'),

    # ── Error pages ────────────────────────────────────────────────────────
    path('403/', views.forbidden_view, name='forbidden'),

    # ── Main (all authenticated) ───────────────────────────────────────────
    path('', views.home_view, name='home'),
    path('klasifikasi/', views.klasifikasi_view, name='klasifikasi'),

    # ── Riwayat personal (semua role) ──────────────────────────────────────
    path('riwayat/', views.riwayat_view, name='riwayat'),
    path('riwayat/hapus/<uuid:pk>/', views.hapus_riwayat, name='hapus_riwayat'),
    path('riwayat/hapus-semua/', views.hapus_semua_riwayat, name='hapus_semua_riwayat'),

    # ── Laporan — download (semua role, konten sesuai role) ────────────────
    path('laporan/', views.laporan_view, name='laporan'),
    path('laporan/cetak/', views.cetak_laporan, name='cetak_laporan'),
    path('laporan/download/pdf/', views.download_laporan_pdf, name='download_pdf'),
    path('laporan/download/excel/', views.download_laporan_excel, name='download_excel'),

    # ── Pengelola & Admin ──────────────────────────────────────────────────
    path('kinerja/', views.kinerja_view, name='kinerja'),

    # ── Admin only ─────────────────────────────────────────────────────────
    path('admin-panel/users/', views.user_management_view, name='user_management'),
    path('admin-panel/users/<uuid:user_id>/role/', views.ubah_role_user, name='ubah_role_user'),
    path('admin-panel/users/<uuid:user_id>/hapus/', views.hapus_user, name='hapus_user'),
    path('admin-panel/users/<uuid:user_id>/edit/', views.edit_user, name='edit_user'),
    path('admin-panel/users/<uuid:user_id>/toggle-active/', views.toggle_active_user, name='toggle_active_user'),
]
