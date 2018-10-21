def predictions(tf,model):
    class_ids = ["Calido,Calmo",
                    "Calido,Calmo,Lluvioso",
                    "Calido,Calmo,Seco",
                    "Calido,Lluvioso",
                    "Calido,Seco",
                    "Calido,Ventoso",
                    "Calido,Ventoso,Lluvioso",
                    "Calido,Ventoso,Seco",
                    "Calmo,Lluvioso",
                    "Caluroso,Calmo",
                    "Caluroso,Calmo,Lluvioso",
                    "Caluroso,Calmo,Seco",
                    "Caluroso,Lluvioso",
                    "Caluroso,Seco",
                    "Caluroso,Ventoso,Lluvioso",
                    "Caluroso,Ventoso,Seco",
                    "Frío",
                    "Frío,Calmo",
                    "Frío,Calmo,Seco",
                    "Frío,Seco",
                    "Frío,Ventoso",
                    "Frío,Ventoso,Lluvioso",
                    "Frío,Ventoso,Seco",
                    "Templado,Calmo",
                    "Templado,Calmo,Lluvioso",
                    "Templado,Calmo,Seco",
                    "Templado,Lluvioso",
                    "Templado,Seco",
                    "Templado,Ventoso",
                    "Templado,Ventoso,Lluvioso",
                    "Templado,Ventoso,Seco"]
            
    predict_dataset = tf.convert_to_tensor([
        [12.8, 6.9, 97.5],#Templado Calmo Lluvioso
        [12.4, 6.5, 55.9],#Calido Ventoso lluvioso
        [6.9, 6, 1],#Frio Calmo Seco
        [-13.0,20.0,200.0 ],#Frio Ventoso lluvioso

    ])
    
    predictions = model(predict_dataset)
    
    for i, logits in enumerate(predictions):
        class_idx = tf.argmax(logits).numpy()
        name = class_ids[class_idx]
        print("Ejemplo {} prediccion: {}".format(i, name))
