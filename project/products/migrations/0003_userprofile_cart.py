# Generated by Django 5.1.3 on 2024-12-14 14:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('products', '0002_productbrand_alter_product_image_alter_product_price'),
    ]

    operations = [
        migrations.AddField(
            model_name='userprofile',
            name='cart',
            field=models.ManyToManyField(to='products.product'),
        ),
    ]
