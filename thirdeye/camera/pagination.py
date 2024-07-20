#camera/pagination.py
from rest_framework.pagination import PageNumberPagination
from .models import Face

class DynamicPageSizePagination(PageNumberPagination):
    def get_page_size(self, request):
        # Dynamically calculate the page size
        total_faces = Face.objects.filter(user=request.user).count()
        page_size = min(total_faces, 100)  # For example, limit to a max of 100 faces per page
        return page_size if page_size > 0 else 10  # Default to 10 if no faces detected
