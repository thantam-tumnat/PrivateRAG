using System.Net.Http;

HttpClient client = new();

string result =
    await client.GetStringAsync(
        "http://127.0.0.1:8000/hello?name=Tan");

Console.WriteLine(result);