import requests
import json

from bs4 import BeautifulSoup


def scrape_instagram(username, person_id):
    # Define the target URL for the Instagram profile
    url = f"https://www.instagram.com/{username}/"

    # Make a request to the Instagram page
    response = requests.get(url)
    if response.status_code != 200:
        print(
            f"Failed to fetch data for {username}. HTTP Status: {response.status_code}"
        )
        return None

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract the script tag containing the shared data
    script_tag = soup.find("script", text=lambda t: t and "window._sharedData" in t)
    if not script_tag:
        print(f"Could not find shared data for {username}")
        return None

    # Extract JSON from the script tag
    try:
        shared_data = json.loads(script_tag.string.split(" = ", 1)[1].rstrip(";"))
        user_data = shared_data["entry_data"]["ProfilePage"][0]["graphql"]["user"]
    except Exception as e:
        print(f"Error parsing data for {username}: {e}")
        return None

    # Build the desired data structure
    structured_data = {
        "id": person_id,
        "name": user_data.get("full_name", "None"),
        "instagram_id": user_data["id"],
        "state": user_data.get("state", False),
        "tags": [
            {
                "value": tag["value"],
                "description": tag["description"],
                "type": tag["type"],
            }
            for tag in user_data.get("tags", "")
        ],
        "instagram": {
            "id": user_data.get("id", ""),
            "url": user_data.get("url", ""),
            "full_name": user_data.get("full_name", ""),
            "bio": user_data.get("biography", ""),
            "follows": user_data.get("edge_follow", {}).get("count", 0),
            "following": user_data.get("edge_followed_by", {}).get("count", 0),
            "timeline_count": user_data.get("edge_owner_to_timeline_media", {}).get(
                "count", 0
            ),
            "posts": [
                {
                    "id": edge.get("idx", ""),
                    "instagram_id": edge.get("id_video", ""),
                    "id_video": edge.get("id_video", ""),
                    "caption": edge.get("edge_media_to_caption", {})
                    .get("edges", [{}])[0]
                    .get("node", {})
                    .get("text", ""),
                    "liked_count": edge.get("edge_liked_by", {}).get("count", 0),
                    "viewed_count": edge.get("video_view_count", 0),
                    "comment_count": edge.get("edge_media_to_comment", {}).get(
                        "count", 0
                    ),
                    "preview_count": edge.get("edge_media_preview_like", {}).get(
                        "count", 0
                    ),
                }
                for idx, edge in enumerate(
                    user_data.get("edge_owner_to_timeline_media", {}).get("edges", [])
                )
            ],
        },
    }

    return structured_data


# Example usage
if __name__ == "__main__":
    username_list = []  # Replace with the Instagram username list

    data = []

    for person_id, username in enumerate(username_list):
        data.append(scrape_instagram(username, person_id))

    if data:
        # Save the structured data to a JSON file
        with open(f"{username}_data.json", "w") as f:
            json.dump(data, f, indent=4)
        print(f"Data saved to {username}_data.json")
