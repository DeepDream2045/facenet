
import unittest
import tensorflow as tf
import numpy as np
import facenet

class DemuxEmbeddingsTest(unittest.TestCase):
  
    def testDemuxEmbeddings(self):
        batch_size = 3*12
        embedding_size = 16
        alpha = 0.2
        
        with tf.Graph().as_default():
        
            embeddings = tf.placeholder(tf.float64, shape=(batch_size, embedding_size), name='embeddings')
            anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1,3,embedding_size]), 3, 1)
            triplet_loss = facenet.triplet_loss(anchor, positive, negative, alpha)
                
            sess = tf.Session()
            with sess.as_default():
                np.random.seed(seed=666)
                emb = np.random.uniform(size=(batch_size, embedding_size))
                tf_triplet_loss = sess.run(triplet_loss, feed_dict={embeddings:emb})

                pos_dist_sqr = np.sum(np.square(emb[0::3,:]-emb[1::3,:]),1)
                neg_dist_sqr = np.sum(np.square(emb[0::3,:]-emb[2::3,:]),1)
                np_triplet_loss = np.mean(np.maximum(0.0, pos_dist_sqr - neg_dist_sqr + alpha))
                
                np.testing.assert_almost_equal(tf_triplet_loss, np_triplet_loss, decimal=5, err_msg='Triplet loss is incorrect')
                      
if __name__ == "__main__":
    unittest.main()
