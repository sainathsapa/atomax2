from django.http import HttpResponseRedirect, HttpResponse

import tweepy
from .img import text_to_image

bearer_token = "AAAAAAAAAAAAAAAAAAAAAMswjQEAAAAAFTVLCk6vSNnUPXUZCs0ncsIE708%3DwUTzX1lrmvELMZ66Or0WM0Yeip0VXJoIKxJceAtzFmzq6IUWhV"

consumer_key = "hTvhz4OtXNbQyXYuuMy5r7VsJ"
consumer_secret = "d8OBXVELlwssy5fRO1x0oOzNU1qVPpqDrK1KWg4S52XFXOV0jL"

access_token = "1592043634374848512-3ypqz1J5WJcFsE7WBSihOUKjTvRiVx"
access_token_secret = "fLV19DXRK8W2f53aOxFLTkJ25VVHZfBi7kMSdA1J378e6"

authenticator = tweepy.OAuthHandler(consumer_key, consumer_secret)
authenticator.set_access_token(access_token, access_token_secret)

api = tweepy.API(authenticator)

client = tweepy.Client(
    consumer_key=consumer_key, consumer_secret=consumer_secret,
    access_token=access_token, access_token_secret=access_token_secret
)


def create_tweet(text, userName):
    response = ""
    if text.count('') > 280:
        return text_withMedia(text, userName)
        # return HttpResponse("Tweet created")
    else:

        response = client.create_tweet(
            text=text
        )
        return response


# print(text_to_image("lih fb sdgfb sdfl sdfbl sdf sdf hsfdhh sdfjhl", "test_atom"))
def text_withMedia(text, userName):

    media = api.media_upload(text_to_image(text, userName))
    reponse = client.create_tweet(
        media_ids=[media.media_id]
    )
    print(text_to_image(text, userName))
    return reponse

#     return HttpResponse(reponse)
