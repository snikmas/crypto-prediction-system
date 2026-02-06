def mapping_coin_data(key: str, data: dict):
    return (
    key, 
    data["time"], 
    data["open"],      
    data["high"],      
    data["low"],       
    data["close"], 
    data["volumefrom"]
)