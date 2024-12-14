from redis import Redis


class RedisClient:
    def __init__(self):
        self.client = Redis(host="redis", port=6379, decode_responses=True)

    def set(self, key, value):
        self.client.set(key, value)

    def get(self, key):
        return self.client.get(key)
    
    def get_all_keys(self):
        return self.client.keys("*")

    def delete(self, key):
        self.client.delete(key)
