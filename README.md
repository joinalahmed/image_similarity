# Image similarity search using deep learning
This project demonstrates, how we can make use of deep learning to do image similarity search. 


## How to run
1. Download [imagenet](https://drive.google.com/open?id=1UOyZ8166qM3SzxGvaUeWpBzSUfoQLUjJ) folder, extract and keep it in  directory
2. Download datasets keep them inside a Dataset directory under server folder. Final folder strcture will be as below
```
      root folder  
      
      │   
      └───server
      |   │───rest-server.py
      |   │───imagenet
      |   │───static
      |   │───templates
      |   │───uploads
      |   │───Dataset
      |        │────Class1
      |              │────product_fold_1
      |                   │────img1.jpg   
      |                   │────img2.jpg  
      |        │────Class2
      |              │────product_fold_1
      |                   │────img1.jpg   
      |                   │────img2.jpg  
               .
               .
               .
               .
               
      |        │────Classn
      |              │────product_fold_1
      |                   │────img1.jpg   
      |                   │────img2.jpg  
      
```
3. Run image vectorizer which passes each data through an inception-v3 model and collects the bottleneck layer vectors and stores in disc. Edit dataset paths accordingly indide the image_vectorizer.py
```
  python server/image_vectorizer.py 
 ```
   This will generate two files namely, image_list.pickle and saved_features.txt. Keep them inside lib folder where search.py script is available.
   
4. Start the server by running rest-server.py. This project uses flask based REST implementation for UI
```
  python server/rest-server.py 
```
5. Once the server starts up, access the url 127.0.0.1:5000 to get the UI. Now upload any file and see 9 similar images. You can change the value of K from 9 to any values, but dont foreget to update the html file accordingly for displaying.

One interesting application of this project is a recommendation engine based on image features. Once the user clicks a product image, the image will go to the server and k-number of similar product images can be displayed on UI as product recommendations. Theses rescommendations are purely based on image similarity. This kind of recommendations have high potentials in fashion-based ecommerce industry.

