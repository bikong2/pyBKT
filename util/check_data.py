def check_data(data):
    #print(data["data"].shape[0])
    #print(data["data"].shape[1])
    #print(data["resources"].shape[0])
    if data["data"].shape[1] != data["resources"].shape[0]: raise IndexError("data and resource sizes must match")
    #print(data["starts"])
    #print(data["lengths"])
    if not all(data["starts"] + data["lengths"] - 1 <= data["data"].shape[1]): raise IndexError("data lengths don't match its shape")
