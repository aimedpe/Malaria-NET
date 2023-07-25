import os
from os.path import join
import argparse
import logging
import pandas as pd
from tqdm import tqdm

from src import functions



def get_args():

    actual = os.getcwd()

    parser = argparse.ArgumentParser()

    parser.add_argument('--patients_dir','-p',type=str,default=join(actual,'data'),metavar=' ',help="File path to the patients' folder in the specified format")
    parser.add_argument('--fastrcnn_falciparum','-ff',type=str,default=join(actual,'models','FastRCNN','FP_FRCNN.pth'),metavar=' ',help='File path to fp_frcnn model')
    parser.add_argument('--fastrcnn_vivax','-fv',type=str,default=join(actual,'models','FastRCNN','VV_FRCNN.pth'),metavar=' ',help='File path to vv_frcnn model')
    parser.add_argument('--resnet_falciparum','-rf',type=str,default=join(actual,'models','ResNet','FP_RESNET.pth'),metavar=' ',help='File path to fp_resnet model')
    parser.add_argument('--resnet_vivax','-rv',type=str,default=join(actual,'models','ResNet','VV_RESNET.pth'),metavar=' ',help='File path to vv_resnet model')
    parser.add_argument('--resnet_last','-rl',type=str,default=join(actual,'models','ResNet','FP_VV_RESNET.pth'),metavar=' ',help='File path to fp_vv_resnet model')
    parser.add_argument('--name_results','-nr',type=str,default="results.csv",metavar=' ',help='File path to the folder where the results will be stored')

    args = parser.parse_args()

    return args


def get_prediction_patient(n_folder: str, args: argparse.Namespace):

    '''
    Calculate the scores for P. falciparum, P. vivax, and uninfected in a single patient.

    Parameters:
    -----------

    n_folder (str): Path to the patient's folder.

    args: Arguments from the file.

    Returns:
    -------

    final_result (str): The diagnostic result of the patient.

    pv_patient_score (float): The score obtained for P. vivax.

    pf_patient_score (float): The score obtained for P. falciparum.

    u_patients_score (float): The score obtained as uninfected.
    '''

    images_patient = os.listdir(n_folder)
    
    total_pv = 0
    total_pf = 0
    sum_u = 0

    for k in tqdm(images_patient):

        k_image = os.path.join(n_folder,k)

        k_prediction_falciparum,k_bboxes_falciparum = functions.predictionFastrcnn(args.fastrcnn_falciparum,k_image)

        k_prediction_falciparum,falciparum_candidates,falciparum_bboxes = functions.predictionResNet(args.resnet_falciparum,2,k_prediction_falciparum,k_bboxes_falciparum)

        logging.info(f"falciparum candidates: {len(falciparum_candidates)}")

        results_falciparum = functions.process_predictionsResNet(k_prediction_falciparum)

        k_prediction_vivax,k_bboxes_vivax = functions.predictionFastrcnn(args.fastrcnn_vivax,k_image)

        k_prediction_vivax,vivax_candidates,vivax_bboxes = functions.predictionResNet(args.resnet_vivax,2,k_prediction_vivax,k_bboxes_vivax)

        logging.info(f"vivax candidates: {len(vivax_candidates)}")

        results_vivax = functions.process_predictionsResNet(k_prediction_vivax)

        duplicate_images,duplicate_bboxes = functions.filtering_duplicates(falciparum_bboxes,falciparum_candidates,
                                                                           vivax_bboxes,vivax_candidates)

        duplicate_probs,_,_ = functions.predictionResNet(args.resnet_last,
                                                     2,
                                                     duplicate_images,
                                                     duplicate_bboxes)
        
        results_duplicates = functions.process_probsResNet_duplicates(duplicate_probs)

        pf,pv = functions.process_results(results_falciparum,results_vivax,results_duplicates)

        logging.info(f"PF:{pf} PV:{pv}")

        if (pf==0) & (pv==1):
            total_pv += results_vivax['n_infected']

        if (pf==0) & (pv==0):
            sum_u += 1

        if (pf==1) & (pv==1):

            k_candidates_probs,_,_ = functions.predictionResNet(args.
                                                              resnet_last,
                                                              2,
                                                              falciparum_candidates+vivax_candidates,
                                                              falciparum_bboxes+vivax_bboxes)
            
            avg_pf,avg_pv = functions.process_probsResNet(k_candidates_probs)

            logging.info(f"avg_pf: {avg_pf} | avg_pv: {avg_pv}")

            if avg_pv > avg_pf: 
                total_pv += results_vivax['n_infected']
            
            else: 
                total_pf += results_falciparum['n_infected']
        
        if (pf==1) & (pv==0):
            total_pf += results_falciparum['n_infected']
        
    
    pv_patient_score = total_pv/len(images_patient)
    pf_patient_score = total_pf/len(images_patient)
    u_patients_score = sum_u/len(images_patient)

    logging.info(f'''
                falciparum patient score: {pf_patient_score}
                vivax patient score: {pv_patient_score}
                uninfected patient score: {u_patients_score}
                ''')

    if u_patients_score>=0.5:

        final_result = 'uninfected'
    
    else:

        if pv_patient_score>pf_patient_score:
            
            final_result = 'vivax'
        
        else:

            final_result = 'falciparum'
    
    logging.info(f"final result: {final_result}")

    return final_result,pv_patient_score,pf_patient_score,u_patients_score

if __name__=='__main__':

    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info("MALARIANET STARTING:...")

    patients_folders = sorted(os.listdir(args.patients_dir))

    patients_name = []
    patients_root = []
    patients_results = []
    patients_pv_score = []
    patients_pf_score = []
    patients_pu_score = []

    logging.info(f"processing {len(patients_folders)} patients...")

    for i,n in enumerate(patients_folders):

        logging.info(f"PATIENT_{i+1}: {n}")

        n_folder = os.path.join(args.patients_dir,n)

        n_prediction,n_pv,n_pf,n_pu = get_prediction_patient(n_folder,args)

        patients_name.append(n)
        patients_root.append(n_folder)
        patients_results.append(n_prediction)
        patients_pv_score.append(n_pv)
        patients_pf_score.append(n_pf)
        patients_pu_score.append(n_pu)
    
    data_results = {
        "patient_name":patients_name,
        "root_images":patients_root,
        "results":patients_results,
        "vivax_score":patients_pv_score,
        "falciparum_score":patients_pf_score,
        "uninfected_score":patients_pu_score
    } 

    results_df = pd.DataFrame(data_results)

    results_path = os.path.join("results",args.name_results)

    logging.info(f"Saving the results in: {results_path}")

    results_df.to_csv(results_path)
    