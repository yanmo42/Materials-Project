import requests; print(requests.get('https://api.openalex.org/works?search=ferroelectric&filter=open_access.is_oa:true&per-page=1').json()['meta']['count'])


print(requests.get('https://api.openalex.org/works?search=ferromagnet&filter=open_access.is_oa:true&per-page=1').json()['meta']['count'])

# 60496 on 7/21/25
# 176482 for ferromagnet/magnetic


