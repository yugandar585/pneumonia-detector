using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using System.Collections;

public class APIManager : MonoBehaviour
{
    public Image overlayImage;
    public Text resultText;

    string url = "http://192.168.0.106:8000/predict";

    public void SendRequest()
    {
        StartCoroutine(CallAPI());
    }

    IEnumerator CallAPI()
    {
        byte[] imgData = System.IO.File.ReadAllBytes("test.jpg");

        WWWForm form = new WWWForm();
        form.AddBinaryData("file", imgData, "test.jpg", "image/jpg");

        UnityWebRequest www = UnityWebRequest.Post(url, form);

        yield return www.SendWebRequest();

        if (www.result == UnityWebRequest.Result.Success)
        {
            string response = www.downloadHandler.text;
            Debug.Log(response);

            if (response.Contains("PNEUMONIA"))
            {
                resultText.text = "PNEUMONIA DETECTED";
                overlayImage.color = new Color(1, 0, 0, 0.3f); // red transparent
            }
            else
            {
                resultText.text = "NORMAL";
                overlayImage.color = new Color(0, 1, 0, 0.3f); // green transparent
            }
        }
        else
        {
            Debug.Log("Error: " + www.error);
        }
    }
}