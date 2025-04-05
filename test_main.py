from  main import predict, data, rf_model

def test_predict():
    # given 
    x = data
    model = rf_model
    
    # when 
    pred_probs = predict(model, x, type="prob")
    
    # then
    assert len(pred_probs) == len(data)