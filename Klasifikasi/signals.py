"""
Signals untuk otomatisasi pembuatan UserProfile.

- Setiap CustomUser baru yang is_superuser=True → role='admin'
- Setiap CustomUser baru lainnya → role='pengguna_studio' (default)
  (Register view tetap membuat UserProfile manual, ini sebagai fallback)
"""
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.conf import settings


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_user_profile(sender, instance, created, **kwargs):
    """Buat UserProfile otomatis saat CustomUser dibuat."""
    if created:
        from .models import UserProfile
        # Superuser (dari createsuperuser) mendapat role admin
        role = 'admin' if instance.is_superuser else 'pengguna_studio'
        UserProfile.objects.get_or_create(
            user=instance,
            defaults={'role': role, 'is_active': True}
        )
