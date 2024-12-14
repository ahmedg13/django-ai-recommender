from django.contrib.auth.models import User
from .models import Product ,UserInteraction ,UserProfile
from django.core.files import File
from pathlib import Path
import random , os , names,datetime

random_brands = {
    "phone":
    {
        "apple":["iphone" ],
        "samsung":["galaxy s" ,"galaxy y"],
        "infinix":["infinix"],
        "realme":["realme"],
        
        
    },
    "laptop":{
        "apple":["macbook"],
        "lenovo":["macbook"],
        "samsung":["macbook"],
        
    },
    "taplet":{
        "apple":["ipad "],
        "samsung":["orpit "],
        "olk":["star "],
        "lenovo":["lenovo A"],
        
    }    
}


def add_users(count=1):
    
    for i in range(count):
        username =names.get_first_name()
        while (User.objects.filter(username=username)):
            username =names.get_first_name()

        User(username=username ,password="123456" ,email=username +"@email.oo",).save()
        print(username +"added")
    


def add_products(count=1):


    for i in range(count):
        catagory = None
        while (  catagory == None or not random_brands[catagory] ):
            catagory = random.choice( list(random_brands.keys()) )

        brand = None
        while (  brand == None or not random_brands[catagory][brand] ):
            brand = random.choice( list(random_brands[catagory].keys()))

        product_name = None
        while (  product_name == None or Product.objects.filter(name=product_name) ):
            product_name = random.choice(random_brands[catagory][brand]) + str(random.randint(0 ,100) )

        img_path ="./Imgs/"+catagory+"/"+random.choice(os.listdir("./Imgs/"+catagory))
        
        Product(
            name=product_name
            ,price=  random.randint(0,5000)+random.random() 
            ,category=catagory
            ,brand=brand
            ,stock=random.randint(0,50 )
            ,image= File(Path(img_path).open(mode="rb") ,name=product_name )

        ).save()

        pass

    pass


def add_interactions(count=1):

    for i in range(count):

        UserInteraction(
            user=random.choice(User.objects.all() ) 
            ,product=random.choice(Product.objects.all())
            ,interaction_type=random.choice(UserInteraction.INTERACTION_TYPES)[0] 
            ,timestamp= datetime.datetime.now()   ).save()
        pass

def create_profiles():
    for u in User.objects.all():
        UserInteraction(user=u)
        



# add_interactions(count=682)
# add_products(200)

