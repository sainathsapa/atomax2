
import json
import requests
import webbrowser
import random
import string
from django.http import HttpResponseRedirect, HttpResponse


from urllib.parse import urlparse, parse_qs

api_url = 'https://www.linkedin.com/oauth/v2'


def authorize(api_url, client_id, client_secret, redirect_uri):

    # Request authentication URL
    csrf_token = create_CSRF_token()
    params = {
        'response_type': 'code',
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'state': csrf_token,
        'scope': 'r_liteprofile,r_emailaddress,w_member_social'
    }
    response = requests.get(f'{api_url}/authorization', params=params)

    print(f'''
    The Browser will open to ask you to authorize the credentials.\n
    Since we have not set up a server, you will get the error:\n
    This site can’t be reached. localhost refused to connect.\n
    This is normal.\n
    You need to copy the URL where you are being redirected to.\n
    ''')

    open_url(response.url)
    redirect_response = input('Paste the full redirect URL here:')
    auth_code = parse_redirect_uri(redirect_response)
    return auth_code


def open_url(url):
    '''
    Function to Open URL.
    Used to open the authorization link
    '''
    print(url)
    webbrowser.open(url)


def read_creds(filename):
    credentials = {}
    with open(filename) as f:
        credentials = json.load(f)
    return credentials


def parse_redirect_uri(redirect_response):

    url = urlparse(redirect_response)
    url = parse_qs(url.query)
    return url['code'][0]


def save_token(filename, data):

    data = json.dumps(data, indent=4)
    with open(filename, 'w') as f:
        f.write(data)


def headers(access_token):

    headers = {
        'Authorization': f'Bearer {access_token}',
        'cache-control': 'no-cache',
        'X-Restli-Protocol-Version': '2.0.0'
    }
    return headers


def create_CSRF_token():

    letters = string.ascii_lowercase
    token = ''.join(random.choice(letters) for i in range(20))
    return token


def refresh_token(auth_code, client_id, client_secret, redirect_uri):

    access_token_url = 'https://www.linkedin.com/oauth/v2/accessToken'
    data = {
        'grant_type': 'authorization_code',
        'code': auth_code,
        'redirect_uri': redirect_uri,
        'client_id': client_id,
        'client_secret': client_secret
    }
    response = requests.post(access_token_url, data=data, timeout=30)
    response = response.json()
    print(response)
    access_token = response['access_token']
    return access_token


def auth(credentials):

    creds = read_creds(credentials)
    print(creds)
    client_id, client_secret = creds['client_id'], creds['client_secret']
    redirect_uri = creds['redirect_uri']
    api_url = 'https://www.linkedin.com/oauth/v2'

    if 'access_token' not in creds.keys():
        args = client_id, client_secret, redirect_uri
        auth_code = authorize(api_url, *args)
        access_token = refresh_token(auth_code, *args)
        creds.update({'access_token': access_token})
        save_token(credentials, creds)
    else:
        access_token = creds['access_token']
        return access_token


credentials = 'files/credentials.json'
access_token = auth(credentials)  # Authenticate the API
headers = headers(access_token)  # Make the headers to attach to the API call.


def user_info(headers):

    response = requests.get('https://api.linkedin.com/v2/me', headers=headers)
    user_info = response.json()
    return user_info


# Get user id to make a UGC post
# user_info = user_info(headers)
# urn = user_info['id']
# print(urn)


def post_to_linkedin(msg, link, link_text):

    api_url = 'https://api.linkedin.com/v2/ugcPosts'
    author = f'urn:li:person:LMKh9zSnbP'

    message = msg
    link = link
    link_text = link_text

    # post_data = {
    #     "author": author,
    #     "lifecycleState": "PUBLISHED",
    #     "specificContent": {
    #         "com.linkedin.ugc.ShareContent": {
    #             "shareCommentary": {
    #                 "text": message
    #             },
    #             "shareMediaCategory": "ARTICLE",
    #             "media": [
    #                 {
    #                     "description": {
    #                         "text": message
    #                         # },
    #                         # "originalUrl": link,
    #                         # "title": {
    #                         #     "text": link_text
    #                     },
    #                     "status": "READY"

    #                 }
    #             ]
    #         }
    #     },

    #     "visibility": {
    #         "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
    #     }
    # }
    post_data = {
        "author": author,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {
                    "text": message
                },
                "shareMediaCategory": "NONE"
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        }
    }
    r = requests.post(api_url, headers=headers, json=post_data)
    return HttpResponse(r)



