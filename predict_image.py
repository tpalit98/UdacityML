def predict(image_path, model, top_k, class_names):
    img = Image.open(image_path)
    test_image = np.asarray(img)
    processed_test_image = process_image(test_image)
    
    prediction = model.predict(np.expand_dims(processed_test_image, axis=0))
    top_values, top_indices = tf.math.top_k(prediction, top_k)
    top_classes = [class_names[str(value)] for value in top_indices.cpu().numpy()[0]]
    top_values_final = top_values.numpy()[0] 
    
    return top_values_final, top_classes