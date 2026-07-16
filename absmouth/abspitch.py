import requests

class AbsolutePitchClient:
    """Client to insert track metadata and embeddings to the Absolute Pitch backend."""
    def __init__(self, config):
        self.api_url = config.API_URL

    def insert_track(self, subsonic_id, title, album_name, artist_id, artist_name, embedding):
        """
        Sends the 1024-d embedding and track metadata to Absolute Pitch.
        Returns the HTTP response.
        """
        payload = {
            "subsonic_id": subsonic_id,
            "title": title,
            "album_name": album_name,
            "artist_id": artist_id,
            "artist_name": artist_name,
            "embedding": embedding
        }
        
        response = requests.post(self.api_url, json=payload)
        return response

    def check_track(self, subsonic_id):
        """
        Queries the Absolute Pitch backend to check if a track has already been processed.
        Returns the HTTP response.
        """
        url = f"{self.api_url.rstrip('/')}/{subsonic_id}"
        return requests.get(url)

