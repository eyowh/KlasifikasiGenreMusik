"""
Decorators dan helper untuk sistem kontrol akses berbasis role.

Roles:
  admin              — akses penuh
  pengelola_studio   — operasional (klasifikasi, semua riwayat, laporan, evaluasi)
  pengguna_studio    — terbatas (upload, riwayat sendiri, download sendiri)

Catatan:
  Cek is_active UserProfile hanya diberlakukan untuk role admin.
  Pengelola studio dan pengguna studio tidak perlu diaktifkan secara manual.
"""

from functools import wraps
from django.shortcuts import redirect


def get_user_role(user):
    """Kembalikan role string dari UserProfile, atau None jika tidak ada profil."""
    if not user.is_authenticated:
        return None
    try:
        return user.profile.role
    except Exception:
        # Fallback: superuser → admin
        if user.is_superuser:
            return 'admin'
        return 'pengguna_studio'


def _is_admin_and_inactive(user):
    """
    Kembalikan True hanya jika user adalah admin dan profilnya
    dinonaktifkan secara eksplisit.
    Cek is_active TIDAK diberlakukan untuk pengelola_studio dan pengguna_studio.
    """
    role = get_user_role(user)
    if role != 'admin':
        return False
    try:
        return not user.profile.is_active
    except Exception:
        return False




def role_required(*roles):
    """
    Decorator: izinkan akses hanya untuk role yang tercantum.
    Jika tidak login → redirect ke /login/
    Jika login tapi role tidak cukup → redirect ke /403/
    """
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return redirect('/login/')
            # Cek is_active hanya untuk admin
            if _is_admin_and_inactive(request.user):
                from django.contrib.auth import logout
                logout(request)
                return redirect('/login/')
            user_role = get_user_role(request.user)
            if user_role not in roles:
                return redirect('/403/')
            return view_func(request, *args, **kwargs)
        return _wrapped
    return decorator


def login_and_active_required(view_func):
    """
    Decorator minimal: hanya memastikan user sudah login.
    is_active tidak dicek untuk pengelola_studio dan pengguna_studio.
    """
    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('/login/')
        # Cek is_active hanya untuk admin
        if _is_admin_and_inactive(request.user):
            from django.contrib.auth import logout
            logout(request)
            return redirect('/login/')
        return view_func(request, *args, **kwargs)
    return _wrapped
