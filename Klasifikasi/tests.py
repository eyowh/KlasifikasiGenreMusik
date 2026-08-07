from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from Klasifikasi.models import UserProfile
from Klasifikasi.views import build_feature_vector, extract_features
import numpy as np
from unittest.mock import patch

User = get_user_model()

class AuthViewsTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='testpassword', email='test@test.com')
        self.profile = UserProfile.objects.create(user=self.user, role='pengguna_studio')

    def test_login_view(self):
        # Test GET request
        response = self.client.get(reverse('Klasifikasi:login'))
        self.assertEqual(response.status_code, 200)

        # Test POST request successful login
        response = self.client.post(reverse('Klasifikasi:login'), {
            'username': 'testuser',
            'password': 'testpassword'
        })
        # Should redirect to home
        self.assertRedirects(response, reverse('Klasifikasi:home'), fetch_redirect_response=False)

        # Test POST request failed login
        response = self.client.post(reverse('Klasifikasi:login'), {
            'username': 'testuser',
            'password': 'wrongpassword'
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Username atau password salah.')

    def test_logout_view(self):
        self.client.login(username='testuser', password='testpassword')
        response = self.client.get(reverse('Klasifikasi:logout'))
        self.assertRedirects(response, reverse('Klasifikasi:login'), fetch_redirect_response=False)

    def test_register_view(self):
        # Test validasi username duplikat
        response = self.client.post(reverse('Klasifikasi:register'), {
            'username': 'testuser',
            'email': 'new@test.com',
            'password': 'newpassword',
            'password_confirm': 'newpassword'
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'sudah digunakan')

        # Test validasi password mismatch
        response = self.client.post(reverse('Klasifikasi:register'), {
            'username': 'newuser',
            'email': 'new@test.com',
            'password': 'newpassword',
            'password_confirm': 'differentpassword'
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'tidak cocok')


class FeatureExtractionTest(TestCase):
    def test_build_feature_vector(self):
        features_dict = {'mfcc_1_mean': 1.0, 'tempo': 120.0}
        vector, names = build_feature_vector(features_dict)
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.shape[0], 1)
        self.assertTrue(len(names) > 0)

    @patch('Klasifikasi.views.librosa')
    def test_extract_features_mock(self, mock_librosa):
        # Setup mock for librosa to avoid needing real audio files during tests
        mock_librosa.feature.mfcc.return_value = np.zeros((13, 100))
        mock_librosa.feature.chroma_stft.return_value = np.zeros((12, 100))
        mock_librosa.feature.spectral_centroid.return_value = np.zeros((1, 100))
        mock_librosa.feature.spectral_bandwidth.return_value = np.zeros((1, 100))
        mock_librosa.feature.spectral_rolloff.return_value = np.zeros((1, 100))
        mock_librosa.feature.zero_crossing_rate.return_value = np.zeros((1, 100))
        mock_librosa.feature.rms.return_value = np.zeros((1, 100))
        mock_librosa.feature.spectral_contrast.return_value = np.zeros((7, 100))
        mock_librosa.feature.tonnetz.return_value = np.zeros((6, 100))
        mock_librosa.beat.beat_track.return_value = (120.0, None)
        mock_librosa.effects.harmonic.return_value = np.zeros(22050)

        with patch('Klasifikasi.views._load_audio_robust', return_value=(np.zeros(22050), 22050)):
            features = extract_features('dummy.wav')
            self.assertIn('mfcc_1_mean', features)
            self.assertIn('tempo', features)
            self.assertEqual(features['tempo'], 120.0)


class RoleAccessTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.admin = User.objects.create_user(username='admin', password='password')
        UserProfile.objects.create(user=self.admin, role='admin')
        
        self.user = User.objects.create_user(username='user', password='password')
        UserProfile.objects.create(user=self.user, role='pengguna_studio')

    def test_role_required(self):
        # Test admin view access by normal user (should be forbidden / redirect to 403)
        self.client.login(username='user', password='password')
        response = self.client.get(reverse('Klasifikasi:user_management'))
        # Usually role_required redirects to forbidden or returns 403
        self.assertTrue(response.status_code in [302, 403])
        if response.status_code == 302:
            self.assertIn('/403/', response.url)

        # Test admin view access by admin
        self.client.login(username='admin', password='password')
        response = self.client.get(reverse('Klasifikasi:user_management'))
        self.assertEqual(response.status_code, 200)

    def test_login_and_active_required(self):
        # Unauthenticated user should be redirected to login
        self.client.logout()
        response = self.client.get(reverse('Klasifikasi:home'))
        self.assertEqual(response.status_code, 302)
        self.assertIn('login', response.url)
