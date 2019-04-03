predictions_orig = model.predict_proba(X_test) 
predictions = model.predict_proba(X_test)
predictions_binary = model.predict(X_test)
for i in range(len(predictions)):
    if predictions[i,0] > predictions[i,1]:
        predictions[i, 0] = 1 - predictions[i,0]
    else:
        predictions[i, 0] = predictions[i,1]
predictions = np.delete(predictions, 1, 1)
