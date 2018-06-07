import logging
import tensorflow as tf
import config
import generate_batch as GB

'''
In main.py, you can finaly run this model
'''


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s :%(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    filename=config["logging_name"],
                    filemode="w"
                    )



sess=tf.Session()
model_1=model(sess=sess,config=config,logging=logging)
model_1.print_var()
for k in range(config["max_epoch"]):
    x_input,y_input=GB(config["batch_size"])
    model_1.train(x_input,y_input,k)
