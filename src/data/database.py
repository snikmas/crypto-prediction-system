# here: connect to the db -> get data from it. do not anything with api / or have to update?
import requests 
from constants import COINS 


data = []
for coin in COINS:



response = requests.get(
    "https://min-api.cryptocompare.com/data/v2/histohour",
    params={"fsym":"BTC","tsym":"USD","limit":"10","api_key":"ea6986b1cbad172494c644a4c88e2c2d6f9cbcb7a0b125dfd4cb8d1451b473a3"},
    headers={"Content-type":"application/json; charset=UTF-8"}
)

json_response = response.json()