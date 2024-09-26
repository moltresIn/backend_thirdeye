from rest_framework import permissions
from authentication.models import UserRole

class IsPaidUser(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user.is_authenticated and request.user_role in [UserRole.PAID, UserRole.TRIAL]

class IsTrialUser(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user.is_authenticated and request.user_role == UserRole.TRIAL
