# ModelDeployment-Xray

### Integration of Machine learning model.

We utilized PyTorch to train our model and saved it in the .pth format, which represents the trained model file. Subsequently, we loaded the trained model and created a POST API endpoint using Flask framework. This endpoint accepts input in the form of a lung X-ray image, processes it, and responds with a JSON format containing the image file and a prediction flag. In this case, a prediction flag of 0 indicates a normal condition, while 1 indicates the presence of pneumonia in the X-ray.

The API endpoint has been integrated into a React application on the frontend. When data is received as a response, it is saved into the database by calling another POST API that was created within the Spring backend application. This ensures the data obtained from the prediction is stored and accessible for further use within the application.
