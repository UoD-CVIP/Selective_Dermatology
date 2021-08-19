# Robust Selective Classification of Skin Lesions with Asymmetric Costs
**Jacob Carse¹, Tamás Süveges¹, Stephen Hogg¹, Emanuele Trucco¹, Charlotte Proby², Colin Fleming³ and Stephen McKenna¹**

¹CVIP, School of Science and Engineering, University of Dundee, Scotland, UK\
²School of Medicine, Ninewells Hospital and Medical School, Dundee, UK\
³Department of Dermatology, Ninewells Hospital and Medical School, Dundee, UK



Automated image analysis of skin lesions has potential to improve diagnostic decision making. A clinically useful system should be selective, rejecting images it is ill-equipped to classify, for example because they are of lesion types not represented well in training data. Furthermore, lesion classifiers should support cost-sensitive decision making. We investigate methods for selective, cost-sensitive classification of lesions as benign or malignant using test images of lesion types represented and not represented in training data. We propose EC-SelectiveNet, a modification to SelectiveNet that discards the selection head at test time, making decisions based on expected costs instead. Experiments show that training for full coverage is beneficial even when operating at lower coverage, and that EC-SelectiveNet outperforms standard cross-entropy training, whether or not temperature scaling or Monte Carlo dropout averaging are used, in both symmetric and asymmetric cost settings.
