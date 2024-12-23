def calculate_lateral_error(data):
    return data["lateral_error"].mean(), data["lateral_error"].std()

def calculate_heading_error(data):
    return data["heading_error"].mean(), data["heading_error"].std()

def calculate_lap_time(data):
    return data["timestamp"].iloc[-1] - data["timestamp"].iloc[0]
