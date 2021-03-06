# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:55:38 2019

@author: Victor Zuanazzi
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:22:26 2019

@author: Victor Zuanazzi
"""
#unit test library
import unittest
import torch
#helpful libraries
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

#import functions to be tested
from data import get_embeddings, create_vocab


class test(unittest.TestCase):

    def test_get_embeddings_from_dict(self):
        """Test bug free value for random predictions and accuracies."""
        
        true_embedding = torch.tensor([-4.2368e-01, -1.6139e-01, -1.6705e-01, -2.2110e-01,  1.5128e-01, 5.6335e-01, -5.5981e-01,  2.5166e-01, -5.5308e-01,  1.5218e+00, -7.6070e-01, -4.0695e-01, -3.5600e-01, -3.5665e-01,  5.6797e-02,
             3.3323e-01, -2.4842e-01,  1.2766e+00,  2.4360e-01, -4.8385e-01,
            -4.4616e-01,  7.1263e-01, -4.4665e-01,  1.5200e-03,  1.2200e-01,
            -6.3202e-01, -1.5182e-01, -1.1545e-01,  6.4300e-02, -3.2581e-01,
            -4.8639e-01,  5.2636e-02,  4.5014e-01,  1.7210e-01, -2.8895e-01,
             6.0675e-01,  1.4416e-01, -6.7070e-01, -4.2227e-01,  2.0163e-01,
            -1.3450e-01, -3.1404e-01, -7.1915e-02,  8.0842e-02,  6.5808e-01,
            -1.1981e-01, -8.0703e-01,  5.3231e-01,  4.2350e-01,  3.7394e-01,
            -4.0781e-01, -4.2753e-01, -2.8070e-01,  8.7740e-02, -3.5463e-01,
            -7.0431e-02,  3.2723e-01,  4.8104e-01, -5.1336e-02,  3.8920e-01,
            -7.4717e-02,  2.8593e-01, -3.7386e-01,  4.7152e-02, -3.4492e-02,
            -2.1526e-01,  3.9959e-02,  1.6208e-01, -3.8727e-01, -1.5322e-01,
            -1.6222e-01, -2.3740e-01, -7.3364e-01,  4.2310e-01, -9.6795e-02,
             7.3406e-04, -3.9757e-01,  1.0300e-01, -4.0056e-01, -3.2450e-01,
            -1.2264e-01,  3.5642e-01,  3.4705e-02,  1.0236e-01,  1.5739e-01,
            -4.4360e-02,  8.2087e-01,  7.0733e-01,  3.1705e-01,  2.6713e-01,
             1.7888e-01,  1.9422e-01, -2.0848e-01, -4.0761e-01, -2.8691e-02,
            -2.6182e-01, -3.0944e-01,  1.2377e-01,  2.4633e-02,  3.4372e-01,
             2.8113e-02, -1.1014e-01, -2.0717e-01,  3.3749e-01, -6.0184e-01,
            -7.7224e-01,  1.0734e-01, -1.5228e-01,  1.5283e-01,  7.2704e-01,
             7.4288e-01,  2.5599e-01,  2.2997e-01,  7.3568e-04,  1.8160e-01,
             2.9630e-02, -1.8097e-01,  1.7985e-01,  3.2480e-01, -1.2208e-01,
            -1.2651e-01, -2.2980e-01,  8.5285e-01, -7.9130e-01, -4.2692e-01,
             2.3197e-01,  1.8495e-01, -3.1808e-01, -4.2642e-01, -1.7526e-01,
            -5.2840e-01, -6.0515e-01,  3.7945e-01, -1.1683e-02,  2.9341e-01,
            -1.5591e-01, -3.1962e-02, -9.9151e-02,  7.8993e-01,  2.8216e-01,
            -2.4709e+00,  8.1358e-01,  1.3619e-01,  4.5629e-01,  7.3905e-02,
             2.4872e-02,  1.6986e-01,  5.3324e-01,  3.4771e-01, -3.9308e-01,
            -3.6146e-01,  3.6111e-02, -3.8388e-01, -2.6158e-01, -2.1516e-01,
            -7.9302e-02,  1.9804e-01,  1.3555e-01,  4.4894e-01, -4.3375e-01,
             6.7353e-01, -2.0403e-01, -2.9590e-01,  9.4962e-03,  2.9652e-01,
             4.5237e-01, -9.7101e-02, -1.7543e-01,  1.7606e-01,  3.3408e-02,
             1.7857e-01,  1.3594e-01,  3.0863e-01, -1.2634e-02, -5.2281e-01,
             1.6341e-01,  4.5821e-01,  1.3761e-02,  3.7040e-01,  3.9126e-01,
            -1.6445e-01,  2.2998e-01,  4.8075e-01, -1.0169e-01,  3.5438e-01,
            -3.3099e-01, -3.1492e-01, -1.8639e-01,  4.3889e-01, -5.5326e-01,
             7.8166e-02,  6.2561e-02,  2.5501e-03, -3.9115e-01, -1.9002e-01,
             3.3724e-01,  6.5801e-02,  5.5904e-01,  5.6657e-01, -1.7535e-01,
            -3.8883e-01,  2.1199e-01, -4.0798e-01,  4.7971e-01,  7.6615e-01,
             1.5916e-01, -1.2325e-01, -1.9361e-01,  4.1964e-01, -6.0248e-02,
             5.3434e-01,  9.1084e-02,  2.4712e-01, -3.0714e-01,  1.3710e-01,
             2.3498e-02, -3.1395e-02, -4.9472e-02, -4.1861e-02, -4.4484e-01,
             5.0607e-01, -3.6262e-01,  7.3686e-01, -3.2171e-01,  1.0322e-02,
             3.2202e-01, -2.8088e-01,  6.8505e-01,  9.3959e-02,  6.4209e-01,
             2.5011e-01,  4.2770e-01, -3.9075e-02,  1.8337e-01,  4.4146e-01,
             1.1847e-01, -1.6311e-01, -3.2149e-02,  2.8192e-01, -2.2174e-01,
            -6.5977e-01,  2.5874e-01,  4.7116e-01, -9.1815e-02, -7.5887e-02,
             3.5397e-01,  1.5350e-01,  4.1432e-01,  7.2722e-03, -3.3496e-01,
            -8.5207e-02,  1.5613e-01, -4.1409e-01,  5.5570e-01,  1.3339e-01,
             6.8005e-02,  1.8927e-01,  1.4143e-01,  3.5294e-01, -3.4540e-01,
             4.7876e-02, -3.1418e-02,  4.9575e-01, -6.4477e-02,  5.0986e-01,
            -8.5942e-02, -1.4068e-01, -1.7043e-01,  1.0659e-01,  1.7795e-01,
            -3.4080e-01, -2.6871e-01,  1.8730e-01, -5.9752e-03,  6.7479e-01,
             1.4230e-01, -2.7056e-01,  3.5657e-01, -1.0204e-01, -8.7691e-02,
             2.2344e-01,  2.7885e-01, -4.5443e-02, -3.0424e-01, -4.9605e-02,
            -2.0646e-01,  2.3481e-01,  1.3190e-01, -5.8691e-01, -2.0702e-02,
            -4.6516e-01, -2.9799e-02, -2.3908e-01, -8.7617e-02,  9.7464e-02,
             4.3836e-01,  9.6314e-03,  3.6152e-01,  2.1664e-03,  1.7918e-01])
        
        word_dict = {'bird': None}
        word_dict = get_embeddings(word_dict)
        epsilon = 1e-7
         
        self.assertLess(sum(true_embedding - word_dict['bird']), epsilon)
    
    def test_get_embeddings_unk(self):
        """Test bug free value for random predictions and accuracies."""
        
        word_dict = {'fkjdsai llds di ***': None} #unknown word.
        
        true_embedding = torch.zeros(300)
        word_dict = get_embeddings(word_dict)

        epsilon = 1e-7
         
        self.assertLess(sum(true_embedding - word_dict['fkjdsai llds di ***']), epsilon)
       
    def test_create_vocab_lenght(self):
       
       sentences = ["I love avocado", 
                     "As mina pira", 
                     "As mina de Pira pira", 
                     "Avocado is overrated"]
       
       #not case sensitive(Avocado and avocado are mapped to avocado)
       vocab = create_vocab(sentences, case_sensitive=False)
       
       #9 unique words + 4 auxiliary words that are always included
       self.assertEqual(13, len(vocab))
             
       #case sensitive (Avocado = Avocado, avocado = avocado)
       vocab = create_vocab(sentences, case_sensitive=True)
       
       #11 unique words + 4 auxiliary words
       self.assertEqual(15, len(vocab))
       

if __name__ == '__main__':
    unittest.main()








