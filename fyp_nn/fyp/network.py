# from keras.models import Sequential
# from keras.layers import Dense
import numpy as np
import pandas as pd


def read_data():
    data_frame_ = pd.DataFrame.from_csv(path="/Users/shoaibanwar/PycharmProjects/neural_test/training_out/training_Characters_30_features.csv")
    return data_frame_


if __name__ == "__main__":
    print("--Network Script Starts----")
    # fix random seed for reproducibility
    np.random.seed(7)

    data_frame = read_data()

    print(data_frame.shape)




    #print(val)

#   Briefly: What is Cloud Infrastructure.
#   Briefly: Network is key element of Cloud Infrastructure
#   Intent of cloud infra is to provide services on demand, that is dynamic
#   Briefly: Dynamic Nature of Cloud Infrastructure (Configuration changes in Network, Topology)
#   Briefly: Related to previous, Easy configuration is desired
#   Saas, Paas, IaaS services are provide:
    # Briefly Describe these and talk about how they utilize/require network dynamicity, changing
#   Briefly: In past, Overlay Networks and many other were used to acquire ease of network configuration
    # Overly Networks
    # Network Virtualization were the solutions.
    # How was Network Virtualization Achieved?
    # Introduce SDN's
    # How does SDN's introduce Network Virtualization
    #