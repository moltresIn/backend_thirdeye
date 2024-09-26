from django.utils import timezone
from authentication.models import Subscription, UserRole

class RoleBasedAccessMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated:
            subscription, _ = Subscription.objects.get_or_create(user=request.user)
            user_role, _ = UserRole.objects.get_or_create(user=request.user)

            if not subscription.is_subscription_active():
                user_role.role = UserRole.UNPAID
                user_role.save()

            request.user_role = user_role.role

        response = self.get_response(request)
        return response
