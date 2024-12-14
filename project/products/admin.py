from django.contrib import admin

from .models import Product ,UserInteraction ,ProductCategory ,ProductBrand

# Register your models here.


admin.site.register(Product)
admin.site.register(UserInteraction)
admin.site.register(ProductCategory)

admin.site.register(ProductBrand)
