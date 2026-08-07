import uuid
from django.conf import settings
from django.db import models
from django.contrib.auth.models import AbstractUser


# ─── Custom User Model ────────────────────────────────────────────────────────

class CustomUser(AbstractUser):
    """Custom User Model dengan UUID sebagai primary key (id_user)."""
    id_user = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    class Meta:
        db_table = 'tb_user'
        verbose_name = 'Pengguna'
        verbose_name_plural = 'Pengguna'

    def __str__(self):
        return self.username


# ─── User Profile (Role System) ──────────────────────────────────────────────

ROLE_CHOICES = [
    ('admin', 'Admin'),
    ('pengelola_studio', 'Pengelola Studio'),
    ('pengguna_studio', 'Pengguna Studio'),
]


class UserProfile(models.Model):
    """Profil pengguna dengan sistem role."""
    id_profile = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='profile',
        db_column='id_user'
    )
    role = models.CharField(max_length=30, choices=ROLE_CHOICES, default='pengguna_studio')
    is_active = models.BooleanField(default=True)
    tanggal_dibuat = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'tb_user_profile'
        verbose_name = 'Profil Pengguna'
        verbose_name_plural = 'Profil Pengguna'

    def __str__(self):
        return f"{self.user.username} [{self.get_role_display()}]"

    def get_role_label(self):
        return dict(ROLE_CHOICES).get(self.role, self.role)


# ─── tb_riwayat_klasifikasi ───────────────────────────────────────────────────

class RiwayatKlasifikasi(models.Model):
    """Model untuk menyimpan riwayat klasifikasi genre musik."""
    id_riwayat = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='riwayat',
        db_column='id_user'
    )
    nama_file = models.CharField(max_length=255)
    ukuran_file = models.CharField(max_length=50)
    format_file = models.CharField(max_length=20)
    genre_prediksi = models.CharField(max_length=100)
    genre_icon = models.CharField(max_length=10, default='🎵')
    confidence = models.FloatField(null=True, blank=True)
    waktu_klasifikasi = models.DateTimeField(auto_now_add=True)
    # Fitur audio teratas sebagai JSON string
    top_features_json = models.TextField(blank=True, default='[]')
    # Distribusi probabilitas sebagai JSON string
    probabilities_json = models.TextField(blank=True, default='[]')

    class Meta:
        db_table = 'tb_riwayat_klasifikasi'
        ordering = ['-waktu_klasifikasi']
        verbose_name = 'Riwayat Klasifikasi'
        verbose_name_plural = 'Riwayat Klasifikasi'

    def __str__(self):
        return f"{self.user.username} -> {self.nama_file} - {self.genre_prediksi}"


# ─── tb_fitur_audio ───────────────────────────────────────────────────────────

class FiturAudio(models.Model):
    """Fitur audio yang diekstrak saat klasifikasi (tb_fitur_audio)."""
    id_fitur_audio = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    id_riwayat = models.OneToOneField(
        RiwayatKlasifikasi,
        on_delete=models.CASCADE,
        related_name='fitur_audio',
        db_column='id_riwayat'
    )
    mfcc_mean = models.FloatField(default=0.0)
    spectral_centroid = models.FloatField(default=0.0)
    zero_crossing_rate = models.FloatField(default=0.0)
    chroma_mean = models.FloatField(default=0.0)
    tempo = models.FloatField(default=0.0)

    class Meta:
        db_table = 'tb_fitur_audio'
        verbose_name = 'Fitur Audio'
        verbose_name_plural = 'Fitur Audio'

    def __str__(self):
        return f"Fitur [{self.id_riwayat.nama_file}]"


# ─── tb_laporan ───────────────────────────────────────────────────────────────

class Laporan(models.Model):
    """Rekaman laporan yang digenerate (tb_laporan)."""
    FORMAT_CHOICES = [
        ('pdf', 'PDF'),
        ('xlsx', 'Excel (XLSX)'),
    ]

    id_laporan = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    id_user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='laporan',
        db_column='id_user'
    )
    judul_laporan = models.CharField(max_length=255)
    periode_awal = models.DateField(null=True, blank=True)
    periode_akhir = models.DateField(null=True, blank=True)
    format_ekspor = models.CharField(max_length=10, choices=FORMAT_CHOICES, default='pdf')
    tanggal_dibuat = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'tb_laporan'
        ordering = ['-tanggal_dibuat']
        verbose_name = 'Laporan'
        verbose_name_plural = 'Laporan'

    def __str__(self):
        return f"{self.judul_laporan} [{self.format_ekspor.upper()}]"
