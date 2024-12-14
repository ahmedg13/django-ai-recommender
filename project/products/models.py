from django.contrib.postgres.fields import ArrayField
from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone


#signals
from django.db.models.signals import post_save 
from django.dispatch import receiver



class Product(models.Model):
    """Represents products in the e-commerce store"""
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2 ,default= 10)
    category = models.CharField(max_length=100 ,null=True)
    brand = models.CharField(max_length=100 ,null=True)
    stock = models.PositiveIntegerField( default=0)
    image = models.ImageField(upload_to='products/', null=True, blank=False)


    def __str__(self):
        return self.name



class ProductBrand(models.Model):
    """Product Brand for more detailed recommendations but for now they do nothing"""
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name


class ProductCategory(models.Model):
    """Product categories for more detailed recommendations but for now they do nothing"""
    name = models.CharField(max_length=100, unique=True)
    parent_category = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL)

    def __str__(self):
        return self.name



@receiver(post_save, sender=Product)
def my_handler(sender,instance ,**kwargs):
    """ function to autmaticly add the brand and category of the product """
    if not ProductCategory.objects.filter(name=instance.category.capitalize()) :
        ProductCategory(name=instance.category.capitalize() ).save()
        
    if not ProductBrand.objects.filter(name=instance.brand.capitalize()) :
        ProductBrand(name=instance.brand.capitalize() ).save()


class UserProfile(models.Model):
    """Extended user profile for personalization"""
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    preferred_categories = models.ManyToManyField(ProductCategory)
    average_spend = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    cart = models.ManyToManyField(Product)

    def __str__(self):
        return f"{self.user}"




class UserInteraction(models.Model):
    """Tracks detailed user interactions with products"""
    INTERACTION_TYPES = [
        ('view', 'Product View'),
        ('cart', 'Added to Cart'),
        ('purchase', 'Purchased'),
        ('wishlist', 'Added to Wishlist'),
        ('review', 'Reviewed Product')
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    interaction_type = models.CharField(max_length=20, choices=INTERACTION_TYPES ,null=True)
    timestamp = models.DateTimeField(default=timezone.now)
    weight = models.FloatField(default=1.0)  # Importance of interaction


    def __str__(self):
        return f"{self.user} {self.interaction_type} {self.product} " 

    class Meta:
        unique_together = ('user', 'product', 'interaction_type')
    

# @receiver(post_save, sender=UserInteraction)
# def _post_save_receiver(sender, instance, **kwargs):
#     if not UserProfile.objects.filter(instance.user):
#         UserProfile(instance.user,).save()


#     pass







