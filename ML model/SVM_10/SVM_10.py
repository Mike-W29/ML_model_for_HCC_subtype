import joblib
import numpy as np
import pandas as pd
import os
# from combat.pycombat import pycombat
from sklearn.metrics import accuracy_score


def data_processing(filepath,
                    log2 = True):

    print('Please note that the input data should be for these 10 genes \n'
          '(ARHGAP11A, CHAF1B, DEPDC1B, ECT2, GINS1, GTSE1, LMNB1, MYBL2, RFC4, TTK), \n'
          'RNA expression TPM values. By default, it will undergo lo2 processing. \n'
          'If the data you input has already undergone log2 processing.\n'
          ' Please select the parameter log2=FALSE')
    # read CSV file
    df = pd.read_csv(filepath,index_col=0)
    Gene = ['ARHGAP11A',
            'CHAF1B',
            'DEPDC1B',
            'ECT2',
            'GINS1',
            'GTSE1',
            'LMNB1',
            'MYBL2',
            'RFC4',
            'TTK']
    df_index_list = df.index.map(str).tolist()

    if  all(elem in df_index_list for elem in Gene): #
        df = df.loc[Gene]
    else:
        raise ValueError("Classification genes were not detected in the data.\n"
                         " Please check if the data contains classification genes")
    df = df.reindex(Gene)
    if log2 == True:
        df = np.log2(df + 1)
    m = np.nanmean(df, axis=1)
    s = np.nanstd(df, axis=1)
    df = (df - m[:, np.newaxis]) / s[:, np.newaxis]
    result = df.T
    return result

def model_prediction(data,outputpath,outpultfilename):
    SVM_model = joblib.load('SVM_model.joblib')
    # The probability of labeling the predicted sample
    y_pred_proba = SVM_model.predict_proba(data)
    # The probability of extracting a prediction as CLASS A and CLASS B
    proba_class_a = y_pred_proba[:, 0]
    proba_class_b = y_pred_proba[:, 1]
    # The labels of the predicted samples
    y_pred_labels = SVM_model.predict(data)
    # Create a DataFrame containing the prediction results
    result_df = pd.DataFrame({
        'Sample': data.index,
        'Probability_ClassA': proba_class_a,
        'Probability_ClassB': proba_class_b,
        'Prediction': y_pred_labels
    })
    # Save the resulting DataFrame to a CSV file
    outputpath = os.path.join(outputpath, outpultfilename) #
    result_df.to_csv(outputpath, index=False)





if __name__ == "__main__":

    input_data_path = '~/GENE_ECXPRESSTION.csv'
    output_path = '~'
    outpultfilename = 'outpult_file_name.csv'
    data = data_processing(filepath=input_data_path,log2=False)
    print(data)
    model_prediction(data,outputpath = output_path,outpultfilename = outpultfilename)