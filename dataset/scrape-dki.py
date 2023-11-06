import requests

with open('scrape-result.csv', 'w', encoding='UTF8') as f:

  for i in range(2017,2024):
    for j in range(1,13):
      url = "https://dashboard-dinkes.jakarta.go.id/rsud/get_diagnosa_all"

      payload = "filter_data=range&start_date="+str(i)+"-"+str(j)+"-01&end_date="+str(i)+"-"+str(j)+"-31"
      headers = {
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Pragma': 'no-cache',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Sec-Fetch-Site': 'same-origin',
        'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
        'Cache-Control': 'no-cache',
        'Sec-Fetch-Mode': 'cors',
        'Accept-Encoding': 'gzip, deflate, br',
        'Origin': 'https://dashboard-dinkes.jakarta.go.id',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
        'Referer': 'https://dashboard-dinkes.jakarta.go.id/rsud',
        'Content-Length': '59',
        'Connection': 'keep-alive',
        'Host': 'dashboard-dinkes.jakarta.go.id',
        'Sec-Fetch-Dest': 'empty',
        'Cookie': 'TS011e8d33=01b53461a6fd083b3cfbf519e7f55f1f040728a56a2975e9e16e65b206d0fb59b0a868f8585598ae2cb8f95741f0228133e10b4cc5350e1825547f32d3a58650a61e13eac1; ci_session_dashboard_dinkes=rdkffdv7ndg63mhmihapgdnm4m91ntg5; TS011e8d33=01b53461a632c4b910f561db4d2e17a03bad5adf4d0e5cd386fc0504d6d1c4808c8253c9e5aed3d6dc05f25bd1d96ce0b92b50fd20217a93e0312681bc8c490c1eae796fb2; ci_session_dashboard_dinkes=rdkffdv7ndg63mhmihapgdnm4m91ntg5',
        'X-Requested-With': 'XMLHttpRequest'
      }

      response = requests.request("POST", url, headers=headers, data=payload)
      # write the data
      print(response.text)
      f.write(str(i)+"-"+str(j)+","+response.text+"\n")
