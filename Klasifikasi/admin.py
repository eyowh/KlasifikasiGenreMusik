from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import (
    CustomUser, UserProfile, RiwayatKlasifikasi,
    FiturAudio, Laporan
)


@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    """Admin panel untuk Custom User Model."""
    model = CustomUser
    list_display = ['id_user', 'username', 'email', 'first_name', 'is_staff', 'is_active']
    search_fields = ['username', 'email', 'first_name']
    ordering = ['username']
    readonly_fields = ['id_user']
    # Tambahkan id_user ke fieldset detail
    fieldsets = UserAdmin.fieldsets + (
        ('ID Unik', {'fields': ('id_user',)}),
    )


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['id_profile', 'user', 'role', 'is_active', 'tanggal_dibuat']
    list_filter = ['role', 'is_active']
    search_fields = ['user__username', 'user__email']
    list_editable = ['role', 'is_active']
    readonly_fields = ['id_profile']


@admin.register(RiwayatKlasifikasi)
class RiwayatKlasifikasiAdmin(admin.ModelAdmin):
    list_display = ['id_riwayat', 'user', 'nama_file', 'genre_prediksi', 'confidence', 'waktu_klasifikasi']
    list_filter = ['genre_prediksi', 'waktu_klasifikasi']
    search_fields = ['user__username', 'nama_file', 'genre_prediksi']
    readonly_fields = ['id_riwayat', 'waktu_klasifikasi']


@admin.register(FiturAudio)
class FiturAudioAdmin(admin.ModelAdmin):
    list_display = ['id_fitur_audio', 'id_riwayat', 'mfcc_mean', 'spectral_centroid', 'zero_crossing_rate', 'chroma_mean', 'tempo']
    search_fields = ['id_riwayat__nama_file']
    readonly_fields = ['id_fitur_audio']


@admin.register(Laporan)
class LaporanAdmin(admin.ModelAdmin):
    list_display = ['id_laporan', 'judul_laporan', 'id_user', 'format_ekspor', 'periode_awal', 'periode_akhir', 'tanggal_dibuat']
    list_filter = ['format_ekspor']
    search_fields = ['judul_laporan', 'id_user__username']
    readonly_fields = ['id_laporan', 'tanggal_dibuat']
