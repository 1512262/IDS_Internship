import pandas as pd
import numpy as np
from PIL import Image
import progressbar
import os 

def get_final_dataframe(split):
        df = pd.read_csv('../data/{}.csv'.format(split))
        flickr30k_path = '../data/flickr30k-images/'
        x_cols=['gt_xmin_1','gt_xmax_1','ddpn_xmin_1','ddpn_xmax_1','gt_xmin_2','gt_xmax_2','ddpn_xmin_2','ddpn_xmax_2']
        y_cols=['gt_ymin_1','gt_ymax_1','ddpn_ymin_1','ddpn_ymax_1','gt_ymin_2','gt_ymax_2','ddpn_ymin_2','ddpn_ymax_2']
        list_file_names = list(set(df.image.tolist()))
        new_df= pd.DataFrame(columns=df.columns)
        i=0
        with progressbar.ProgressBar(max_value=len(list_file_names)) as bar:
                for fname in list_file_names:
                        addr = flickr30k_path+str(fname)+'.jpg'
                        img = Image.open(addr)
                        w,h= img.size
                        temp = df[df.image==fname].copy()
                        temp.loc[:,x_cols]=temp.loc[:,x_cols]* 224.0 / w
                        temp.loc[:,y_cols]=temp.loc[:,y_cols]* 224.0 / h
                        new_df=new_df.append(temp,ignore_index=True)
                        i+=1
                        bar.update(i)
        
        return new_df


def main():
        if not os.path.exists('../data/old_VGP/'):
                os.makedirs('../data/old_VGP/')
        for split in ['train','val','test']:
                print('{} - final_dataframe'.format(split))
                df = get_final_dataframe(split)
                df.to_csv('../data/old_VGP/{}_new.csv'.format(split))

if __name__ == "__main__":
        main()
