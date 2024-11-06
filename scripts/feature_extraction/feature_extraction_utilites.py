import numpy as np
import rasterio
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.stats import kurtosis, skew, entropy
from skimage.feature import graycomatrix, graycoprops
import itertools
from tqdm import tqdm
import cv2
import joblib
import os



epsilon = 1e-6
def calculate_statistics(basic_bands, df, bands):
    
    new_cols = {}
    for j, band in enumerate(tqdm(basic_bands)):
        
        new_cols[str(band) + '_mean'] = np.mean(bands[:,j,:],axis=1)
        new_cols[str(band) + '_var'] = np.var(bands[:,j,:],axis=1)
        new_cols[str(band) + '_median'] = np.median(bands[:,j,:],axis=1)
        new_cols[str(band) + '_kurtosis'] = kurtosis(bands[:,j,:],axis=1)
        new_cols[str(band) + '_skew'] = skew(bands[:,j,:],axis=1)
        new_cols[str(band) + '_std'] = np.std(bands[:,j,:],axis=1)
        new_cols[str(band) + '_p10'] = np.percentile(bands[:,j,:], 10,axis=1)
        new_cols[str(band) + '_p25'] = np.percentile(bands[:,j,:], 25,axis=1)
        new_cols[str(band) + '_p75'] = np.percentile(bands[:,j,:], 75,axis=1)
        new_cols[str(band) + '_p90'] = np.percentile(bands[:,j,:],90,axis=1)
    
    for (i, j) in tqdm(itertools.combinations(range(len(basic_bands)), 2)):  
            Ba = bands[:,i,:]
            Bb = bands[:,j,:]
            new_band = (Ba - Bb) / (Ba + Bb + epsilon)

            new_cols[f'band_{i}_{j}_mean'] = (np.mean(new_band))
            new_cols[f'band_{i}_{j}_median'] = (np.median(new_band))
            new_cols[f'band_{i}_{j}_var'] = (np.var(new_band))
            new_cols[f'band_{i}_{j}_kurtosis'] = kurtosis(new_band,axis=1)
            new_cols[f'band_{i}_{j}_skew'] = skew(new_band,axis=1)
            new_cols[f'band_{i}_{j}_std'] = (np.std(new_band))
            new_cols[f'band_{i}_{j}_10th_percentile'] = (np.percentile(new_band, 10,axis=1))
            new_cols[f'band_{i}_{j}_25th_percentile'] = (np.percentile(new_band, 25,axis=1))
            new_cols[f'band_{i}_{j}_75th_percentile'] = (np.percentile(new_band, 75,axis=1))
            new_cols[f'band_{i}_{j}_90th_percentile'] = (np.percentile(new_band, 90,axis=1))
        
    for (i, j) in tqdm(itertools.combinations(range(len(basic_bands)), 2)):  
            Ba = bands[:,i,:]
            Bb = bands[:,j,:]
            new_band = (Ba) / (Bb + epsilon)

            new_cols[f'band_{i}_{j}_mean'] = (np.mean(new_band))
            new_cols[f'band_{i}_{j}_median'] = (np.median(new_band))
            new_cols[f'band_{i}_{j}_var'] = (np.var(new_band))
            new_cols[f'band_{i}_{j}_kurtosis'] = kurtosis(new_band,axis=1)
            new_cols[f'band_{i}_{j}_skew'] = skew(new_band,axis=1)
            new_cols[f'band_{i}_{j}_std'] = (np.std(new_band))
            new_cols[f'band_{i}_{j}_10th_percentile'] = (np.percentile(new_band, 10,axis=1))
            new_cols[f'band_{i}_{j}_25th_percentile'] = (np.percentile(new_band, 25,axis=1))
            new_cols[f'band_{i}_{j}_75th_percentile'] = (np.percentile(new_band, 75,axis=1))
            new_cols[f'band_{i}_{j}_90th_percentile'] = (np.percentile(new_band, 90,axis=1))
    
    
    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
    
    return df

def calculate_advanced_band_features(basic_bands, df, bands, epsilon=1e-9):

    new_cols = {}
    
    for (i, j) in tqdm(itertools.combinations(range(len(basic_bands)), 2)):
        Ba = bands[:, i, :]
        Bb = bands[:, j, :]
        
        new_cols[f'band_{i}_{j}_geometric_mean'] = np.sqrt(Ba * Bb).mean(axis=1)
        
        new_cols[f'band_{i}_{j}_harmonic_mean'] = (2 * Ba * Bb / (Ba + Bb + epsilon)).mean(axis=1)
        
        new_cols[f'band_{i}_{j}_normalized_ratio'] = ((Ba - Bb) / (np.sqrt(Ba**2 + Bb**2) + epsilon)).mean(axis=1)
    
    for i, band in tqdm(enumerate(basic_bands)):
        Ba = bands[:, i, :]

        new_cols[f'band_{i}_entropy'] = np.apply_along_axis(lambda x: entropy(np.histogram(x, bins=256, density=True)[0]), 1, Ba)

    for (i, j, k) in tqdm(itertools.combinations(range(len(basic_bands)), 3)):
        Ba = bands[:, i, :]
        Bb = bands[:, j, :]
        Bc = bands[:, k, :]
        
        new_cols[f'band_{i}_{j}_{k}_diff_prod'] = ((Ba * Bb) - (Ba * Bc)).mean(axis=1)

    for (i, j, k) in tqdm(itertools.combinations(range(len(basic_bands)), 3)):
        Ba = bands[:, i, :]
        Bb = bands[:, j, :]
        Bc = bands[:, k, :]
        
        new_cols[f'band_{i}_{j}_{k}_custom_ratio'] = ((Ba * Bb) / (Bc + epsilon)).mean(axis=1)

    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
    
    return df


def compute_texture_features(basic_bands,df,bands):

    new_cols = {band: {f'{band}_contrast': [], f'{band}_correlation': [], 
                       f'{band}_energy': [], f'{band}_homogeneity': [], f'{band}_GLCM_Dissimilarity': [],f'{band}_Homogeneity':[]}
                for band in basic_bands}
    for band in basic_bands:
        print(f'{band}_GLCM_Dissimilarity')
    for i in range(bands.shape[0]):

        for j, band in enumerate(basic_bands):

            reshaped_band = bands[i, j, :].reshape(128, 128)
            
            glcm = graycomatrix(reshaped_band.astype(np.uint8), distances=[1], angles=[0], symmetric=True, normed=True)
            new_cols[band][f'{band}_GLCM_Dissimilarity'].append(graycoprops(glcm, 'dissimilarity')[0, 0])
            new_cols[band][f'{band}_Homogeneity'].append(graycoprops(glcm, 'homogeneity')[0, 0])
            new_cols[band][f'{band}_contrast'].append(graycoprops(glcm, 'contrast')[0, 0])
            new_cols[band][f'{band}_correlation'].append(graycoprops(glcm, 'correlation')[0, 0])
            new_cols[band][f'{band}_energy'].append(graycoprops(glcm, 'energy')[0, 0])
            new_cols[band][f'{band}_homogeneity'].append(graycoprops(glcm, 'homogeneity')[0, 0])
    
    for band in basic_bands:
        df_band = pd.DataFrame(new_cols[band])
        df = pd.concat([df, df_band], axis=1)
    
    return df
    
def calculate_vegetation_indices(df, bands):

        new_cols = {}
        B1 = bands[:, 0, :]
        B2 = bands[:, 1, :]
        B3 = bands[:, 2, :]
        B4 = bands[:, 3, :]
        B5 = bands[:, 4, :]
        B6 = bands[:, 5, :]
        B7 = bands[:, 6, :]
        B8 = bands[:, 7, :]
        B8A = bands[:, 8, :]
        B9 = bands[:, 9, :]
        B11 = bands[:, 10, :]
        B12 = bands[:, 11, :]
        
        ndvi = (B8 - B4) / (B8 + B4 + epsilon)
        
        gli = (2 * B3 - B4 - B2) / (2 * B3 + B4 + B2 + epsilon)
        
        cvi = (B8 * B4) / (B3**2 + epsilon)  
        
        sipi = (B8 - B2) / (B8 - B4 + epsilon) 
        
        ndwi = (B8 - B11) / (B8 + B11 + epsilon)  
    
        ccci = (B8 - B5) / (B8 + B5 + epsilon) / (B8 - B4 + epsilon) / (B8 + B4 + epsilon)
        
        hue = np.arctan2((B2 - B3), (B3 - B4))
        
        rendvi = (B6 - B5) / (B6 + B5 + epsilon)
        
        reci = (B8 / (B4 + epsilon)) - 1
        
        evi = 2.5 * (B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1.0 + epsilon)
        
        evi2 = 2.4 * (B8 - B4) / (B8 + B4 + 1.0 + epsilon)
        
        bsi = (B11 + B4) / (B8 + B2 + epsilon)  
    
        npcri = (B3 - B8) / (B3 + B8 + epsilon)
        
        mndvi = (B8 - B4) / (B8 + B4 - 2 * B2 + epsilon)
        
        ndsi = (B3 - B11) / (B3 + B11 + epsilon)
        
        ndmi = (B8 - B11) / (B8 + B11 + epsilon)
        
        ndbi = (B11 - B8) / (B11 + B8 + epsilon)
        
        ndci = (B5 - B4) / (B5 + B4 + epsilon)
        
        gndvi = (B8 - B3) / (B8 + B3 + epsilon)
        
        L = 0.428
        savi = ((B1 - B2) / (B1 + B2 + L + epsilon)) * (1.0 + L)
        
        #####
        vegetation_bands = {
            'ndvi': ndvi, 'gli': gli, 'cvi': cvi, 'sipi': sipi, 'ccci': ccci,
            'hue': hue, 'rendvi': rendvi, 'reci': reci, 'evi': evi, 'evi2': evi2, 'ndwi': ndwi,
            'npcri': npcri, 'mndvi': mndvi, 'ndsi': ndsi, 'ndmi': ndmi, 'ndbi': ndbi, 'ndci': ndci,
            'gndvi': gndvi, 'savi': savi, 'bsi': bsi
        }


        glcm_features = {f'{name}_GLCM_{prop}': [] for name in vegetation_bands for prop in ['Dissimilarity', 'Homogeneity', 'Contrast', 'Correlation', 'Energy']}
        for vegetation_band_name , vegetation_band_vals in tqdm(vegetation_bands.items()):
            
            new_cols[str(vegetation_band_name) + '_mean'] = np.mean(vegetation_band_vals,axis=1)
            new_cols[str(vegetation_band_name) + '_mean'] = np.mean(vegetation_band_vals,axis=1)
            new_cols[str(vegetation_band_name) + '_var'] = np.var(vegetation_band_vals,axis=1)
            new_cols[str(vegetation_band_name) + '_median'] = np.median(vegetation_band_vals,axis=1)
            new_cols[str(vegetation_band_name) + '_kurtosis'] = kurtosis(vegetation_band_vals,axis=1)
            new_cols[str(vegetation_band_name) + '_skew'] = skew(vegetation_band_vals,axis=1)
            new_cols[str(vegetation_band_name) + '_std'] = np.std(vegetation_band_vals,axis=1)
            new_cols[str(vegetation_band_name) + '_p10'] = np.percentile(vegetation_band_vals, 10,axis = 1)
            new_cols[str(vegetation_band_name) + '_p25'] = np.percentile(vegetation_band_vals, 25, axis = 1)
            new_cols[str(vegetation_band_name) + '_p75'] = np.percentile(vegetation_band_vals, 75 , axis = 1)
            new_cols[str(vegetation_band_name) + '_p90'] = np.percentile(vegetation_band_vals,90 , axis = 1)
            for i in range(bands.shape[0]):
        
                reshaped_band = vegetation_band_vals[i].reshape(128, 128)
                
                glcm = graycomatrix(reshaped_band.astype(np.uint8), distances=[1], angles=[0], symmetric=True, normed=True)
        
                glcm_features[f'{vegetation_band_name}_GLCM_Dissimilarity'].append(graycoprops(glcm, 'dissimilarity')[0, 0])
                glcm_features[f'{vegetation_band_name}_GLCM_Homogeneity'].append(graycoprops(glcm, 'homogeneity')[0, 0])
                glcm_features[f'{vegetation_band_name}_GLCM_Contrast'].append(graycoprops(glcm, 'contrast')[0, 0])
                glcm_features[f'{vegetation_band_name}_GLCM_Correlation'].append(graycoprops(glcm, 'correlation')[0, 0])
                glcm_features[f'{vegetation_band_name}_GLCM_Energy'].append(graycoprops(glcm, 'energy')[0, 0])

        all_new_features = pd.DataFrame({**new_cols, **glcm_features})
        df = pd.concat([df.reset_index(drop=True), all_new_features.reset_index(drop=True)], axis=1)
        return df

def calculate_vegetation_indices2(df, bands):
        new_cols = {}
        B1 = bands[:,0,:]
        B2 = bands[:,1,:]
        B3 = bands[:,2,:]
        B4 = bands[:,3,:]
        B5 = bands[:,4,:]
        B6 = bands[:,5,:]
        B7 = bands[:,6,:]
        B8 = bands[:,7,:]
        B8A = bands[:,8,:]
        B9 = bands[:,9,:]
        B11 = bands[:,10,:]
        B12 = bands[:,11,:]
    
        ndvi_r = (B8 - B7) / (B8 + B7 + epsilon)  
        
        chl = (B7 / (B5 + epsilon)) - 1  
        
        bi = (B4**2 + B3**2 + B2**2) / 3 
        
        si = (B4 - B2) / (B4 + B2 + epsilon)  
        
        nmdi = (B8 - (B11 - B12)) / (B8 + (B11 - B12) + epsilon)  
        
        wbi = B8A / (B9 + epsilon)  
        
        msi = B11 / (B8 + epsilon)  
        
        bsi1 = ((B12 + B4) - (B8A + B2)) / ((B12 + B4) + (B8A + B2) + epsilon)
        
        bsi2 = ((B11 + B4) - (B8A + B2)) / ((B11 + B4) + (B8A + B2) + epsilon) 
        
        ndsi1 = (B11 - B8A) / (B11 + B8A + epsilon)  
        
        ndsi2 = (B12 - B3) / (B12 + B3 + epsilon)  
        
        bi2 = B4 + B11 - B8A  
        
        dbsi = ndsi2 - ((B8A - B4) / (B8A + B4 + epsilon))  
        
        mbi = ((B11 + B12 + B8A) / (B11 + B12 + B8A + epsilon)) + 0.5  
        
        r01 = B1 / (B3 + epsilon)  
        
        r02 = B1 / (B5 + epsilon) 
        
        r03 = B11 / (B12 + epsilon)  
    
        r04 = B5 / (B4 + epsilon)  
        
        mi = (B8A - B11) / (B8A + B11 + epsilon)  
        
        mresr = (B6 - B1) / (B5 - B1 + epsilon)  
        
        psri = (B4 - B2) / (B6 + epsilon)  
        
        tvi = (120 * (B6 - B3) - 200 * (B4 - B3)) / 2  
        
        arvi = (B8 - 2 * B4 + B2) / (B8 + 2 * B4 + B2 + epsilon)  
        
        exg = 2 * B3 - B4 - B2  
        
        aci = B8 * (B4 + B3)  
        
        #####
        vegetation_bands = {
            'ndvi_r': ndvi_r, 'chl': chl, 'bi': bi, 'si': si, 'nmdi': nmdi, 'wbi': wbi, 'msi': msi,
            'bsi1': bsi1, 'bsi2': bsi2, 'ndsi1': ndsi1, 'ndsi2': ndsi2, 'bi2': bi2, 'dbsi': dbsi,
            'mbi': mbi, 'r01': r01, 'r02': r02, 'r03': r03, 'r04': r04, 'mi': mi, 'mresr': mresr,
            'psri': psri, 'tvi': tvi, 'arvi': arvi, 'exg': exg, 'aci': aci
        }
        glcm_features = {f'{name}_GLCM_{prop}': [] for name in vegetation_bands for prop in ['Dissimilarity', 'Homogeneity', 'Contrast', 'Correlation', 'Energy']}
        for vegetation_band_name , vegetation_band_vals in tqdm(vegetation_bands.items()):
            
            new_cols[str(vegetation_band_name) + '_mean'] = np.mean(vegetation_band_vals,axis=1)
            new_cols[str(vegetation_band_name) + '_mean'] = np.mean(vegetation_band_vals,axis=1)
            new_cols[str(vegetation_band_name) + '_var'] = np.var(vegetation_band_vals,axis=1)
            new_cols[str(vegetation_band_name) + '_median'] = np.median(vegetation_band_vals,axis=1)
            new_cols[str(vegetation_band_name) + '_kurtosis'] = kurtosis(vegetation_band_vals,axis=1)
            new_cols[str(vegetation_band_name) + '_skew'] = skew(vegetation_band_vals,axis=1)
            new_cols[str(vegetation_band_name) + '_std'] = np.std(vegetation_band_vals,axis=1)
            new_cols[str(vegetation_band_name) + '_p10'] = np.percentile(vegetation_band_vals, 10,axis = 1)
            new_cols[str(vegetation_band_name) + '_p25'] = np.percentile(vegetation_band_vals, 25, axis = 1)
            new_cols[str(vegetation_band_name) + '_p75'] = np.percentile(vegetation_band_vals, 75 , axis = 1)
            new_cols[str(vegetation_band_name) + '_p90'] = np.percentile(vegetation_band_vals,90 , axis = 1)
            for i in range(bands.shape[0]):
        
                reshaped_band = vegetation_band_vals[i].reshape(128, 128)
                
                glcm = graycomatrix(reshaped_band.astype(np.uint8), distances=[1], angles=[0], symmetric=True, normed=True)
        
                glcm_features[f'{vegetation_band_name}_GLCM_Dissimilarity'].append(graycoprops(glcm, 'dissimilarity')[0, 0])
                glcm_features[f'{vegetation_band_name}_GLCM_Homogeneity'].append(graycoprops(glcm, 'homogeneity')[0, 0])
                glcm_features[f'{vegetation_band_name}_GLCM_Contrast'].append(graycoprops(glcm, 'contrast')[0, 0])
                glcm_features[f'{vegetation_band_name}_GLCM_Correlation'].append(graycoprops(glcm, 'correlation')[0, 0])
                glcm_features[f'{vegetation_band_name}_GLCM_Energy'].append(graycoprops(glcm, 'energy')[0, 0])

        all_new_features = pd.DataFrame({**new_cols, **glcm_features})
        df = pd.concat([df.reset_index(drop=True), all_new_features.reset_index(drop=True)], axis=1)
        return df
    

def bloom_indices(df, bands):

    new_cols = {}
    
    B2 = bands[:,1,:]
  
    B3 = bands[:,2,:]
  
    B4 = bands[:,3,:]
        
    B5 = bands[:,4,:]
           
    B6 = bands[:,5,:]
      
    B7 = bands[:,6,:]
       
    B8 = bands[:,7,:]
           
    B8A = bands[:,8,:]
      
    B9 = bands[:,9,:]
    
    B11 = bands[:,10,:]
    
    B12 = bands[:,11,:]

    
    NDGI =  (B4 - B3 )  / (B4 + B3 + epsilon)
    DYI =  B4  / (B3 + epsilon)
    NDPI =  (0.5*(B4 + B2) - B3)  / (0.5*(B4 + B2) + B3 + epsilon)
    PEBI =  NDPI / ((NDGI +1) * B8 + epsilon)
    NDYI =  (0.5*(B4 + B3) - B2)  / (0.5*(B4 + B3) + B2 + epsilon)
    YEBI =  NDYI / ((NDGI +1) * B8 + epsilon) 

    bloom = {'NDGI': NDGI, 'DYI': DYI, 'NDPI': NDPI, 'PEBI': PEBI, 'NDYI': NDYI, 'YEBI': YEBI }
    glcm_features = {f'{name}_GLCM_{prop}': [] for name in bloom for prop in ['Dissimilarity', 'Homogeneity', 'Contrast', 'Correlation', 'Energy']}
    for bloom_indices_name , bloom_indices_vals in tqdm(bloom.items()):
            
        new_cols[str(bloom_indices_name) + '_mean'] = np.mean(bloom_indices_vals,axis=1)
        new_cols[str(bloom_indices_name) + '_mean'] = np.mean(bloom_indices_vals,axis=1)
        new_cols[str(bloom_indices_name) + '_var'] = np.var(bloom_indices_vals,axis=1)
        new_cols[str(bloom_indices_name) + '_median'] = np.median(bloom_indices_vals,axis=1)
        new_cols[str(bloom_indices_name) + '_kurtosis'] = kurtosis(bloom_indices_vals,axis=1)
        new_cols[str(bloom_indices_name) + '_skew'] = skew(bloom_indices_vals,axis=1)
        new_cols[str(bloom_indices_name) + '_std'] = np.std(bloom_indices_vals,axis=1)
        new_cols[str(bloom_indices_name) + '_p10'] = np.percentile(bloom_indices_vals, 10,axis = 1)
        new_cols[str(bloom_indices_name) + '_p25'] = np.percentile(bloom_indices_vals, 25, axis = 1)
        new_cols[str(bloom_indices_name) + '_p75'] = np.percentile(bloom_indices_vals, 75 , axis = 1)
        new_cols[str(bloom_indices_name) + '_p90'] = np.percentile(bloom_indices_vals,90 , axis = 1)

        for i in range(bands.shape[0]):
        
            reshaped_band = bloom_indices_vals[i].reshape(128, 128)
            
            glcm = graycomatrix(reshaped_band.astype(np.uint8), distances=[1], angles=[0], symmetric=True, normed=True)
    
            glcm_features[f'{bloom_indices_name}_GLCM_Dissimilarity'].append(graycoprops(glcm, 'dissimilarity')[0, 0])
            glcm_features[f'{bloom_indices_name}_GLCM_Homogeneity'].append(graycoprops(glcm, 'homogeneity')[0, 0])
            glcm_features[f'{bloom_indices_name}_GLCM_Contrast'].append(graycoprops(glcm, 'contrast')[0, 0])
            glcm_features[f'{bloom_indices_name}_GLCM_Correlation'].append(graycoprops(glcm, 'correlation')[0, 0])
            glcm_features[f'{bloom_indices_name}_GLCM_Energy'].append(graycoprops(glcm, 'energy')[0, 0])
    all_new_features = pd.DataFrame({**new_cols, **glcm_features})
    df = pd.concat([df.reset_index(drop=True), all_new_features.reset_index(drop=True)], axis=1)
    
    return df

def rededge_indices(df, bbands):

    new_cols = {}
    
    B2 = bbands[:,1,:]
  
    B3 = bbands[:,2,:]
  
    B4 = bbands[:,3,:]
        
    B5 = bbands[:,4,:]
           
    B6 = bbands[:,5,:]
      
    B7 = bbands[:,6,:]
       
    B8 = bbands[:,7,:]
           
    B8A = bbands[:,8,:]
      
    B9 = bbands[:,9,:]
    
    B11 = bbands[:,10,:]
    
    B12 = bbands[:,11,:]

    
    
    NDVIre1 =  (B8 - B5)  / (B8 + B5 + epsilon)
    NDVIre2 =  (B8 - B6)  / (B8 + B6 + epsilon)
    NDVIre3 =  (B8 - B7)  / (B8 + B7 + epsilon)

    NDRE1 =  (B6 - B5)  / (B6 + B5 + epsilon)
    NDRE2 =  (B7 - B5)  / (B7 + B5 + epsilon)
    NDRE3 =  (B7 - B6)  / (B7 + B6 + epsilon)

    CIre1 =  (B8 /(B5 + epsilon))  - 1 
    CIre2 =  (B8 /(B6 + epsilon))  - 1
    CIre3 =  (B8 /(B7 + epsilon))  - 1

    MCARI1 =  ((B5 - B4) - 0.2*(B5 - B3)) * (B5 / (B4 + epsilon))
    MCARI2 =  ((B6 - B4) - 0.2*(B6 - B3)) * (B6 / (B4 + epsilon + epsilon))
    MCARI3 =  ((B7 - B4) - 0.2*(B7 - B3)) * (B7 / (B4 + epsilon))

    
    TCARI1 =  3*((B5 - B4) - 0.2*(B5 - B3)) * (B5 / (B4 + epsilon))
    TCARI2 =  3*((B6 - B4) - 0.2*(B6 - B3)) * (B6 / (B4 + epsilon))
    TCARI3 =  3*((B7 - B4) - 0.2*(B7 - B3)) * (B7 / (B4 + epsilon))

    MTCI1 =  (B6 - B5)  / (B5 - B4 + epsilon)
    MTCI2 =  (B7 - B5)  / (B5 - B4 + epsilon)
    MTCI3 =  (B7 - B6)  / (B6 - B4 + epsilon)  
    
    rededge = {'NDVIre1': NDVIre1, 'NDVIre2': NDVIre2, 'NDVIre3': NDVIre3, 'NDRE1': NDRE1, 'NDRE2': NDRE2, 'NDRE3': NDRE3,
            'CIre1': CIre1, 'CIre2': CIre2, 'CIre3': CIre3, 'MCARI1': MCARI1, 'MCARI2': MCARI2, 'MCARI3': MCARI3,
             'TCARI1': TCARI1, 'TCARI2': TCARI2, 'TCARI3': TCARI3, 'MTCI1': MTCI1, 'MTCI2': MTCI2, 'MTCI3': MTCI3,
            }
    glcm_features = {f'{name}_GLCM_{prop}': [] for name in rededge for prop in ['Dissimilarity', 'Homogeneity', 'Contrast', 'Correlation', 'Energy']}
    for rededge_indices_name , rededge_indices_vals in tqdm(rededge.items()):
            
        new_cols[str(rededge_indices_name) + '_mean'] = np.mean(rededge_indices_vals,axis=1)
        new_cols[str(rededge_indices_name) + '_mean'] = np.mean(rededge_indices_vals,axis=1)
        new_cols[str(rededge_indices_name) + '_var'] = np.var(rededge_indices_vals,axis=1)
        new_cols[str(rededge_indices_name) + '_median'] = np.median(rededge_indices_vals,axis=1)
        new_cols[str(rededge_indices_name) + '_kurtosis'] = kurtosis(rededge_indices_vals,axis=1)
        new_cols[str(rededge_indices_name) + '_skew'] = skew(rededge_indices_vals,axis=1)
        new_cols[str(rededge_indices_name) + '_std'] = np.std(rededge_indices_vals,axis=1)
        new_cols[str(rededge_indices_name) + '_p10'] = np.percentile(rededge_indices_vals, 10,axis = 1)
        new_cols[str(rededge_indices_name) + '_p25'] = np.percentile(rededge_indices_vals, 25, axis = 1)
        new_cols[str(rededge_indices_name) + '_p75'] = np.percentile(rededge_indices_vals, 75 , axis = 1)
        new_cols[str(rededge_indices_name) + '_p90'] = np.percentile(rededge_indices_vals,90 , axis = 1)

        for i in range(rededge_indices_vals.shape[0]):
            reshaped_band = rededge_indices_vals[i].reshape(128, 128).astype(np.uint8)
            glcm = graycomatrix(reshaped_band, distances=[1], angles=[0], symmetric=True, normed=True)
            
            glcm_features[f'{rededge_indices_name}_GLCM_Dissimilarity'].append(graycoprops(glcm, 'dissimilarity')[0, 0])
            glcm_features[f'{rededge_indices_name}_GLCM_Homogeneity'].append(graycoprops(glcm, 'homogeneity')[0, 0])
            glcm_features[f'{rededge_indices_name}_GLCM_Contrast'].append(graycoprops(glcm, 'contrast')[0, 0])
            glcm_features[f'{rededge_indices_name}_GLCM_Correlation'].append(graycoprops(glcm, 'correlation')[0, 0])
            glcm_features[f'{rededge_indices_name}_GLCM_Energy'].append(graycoprops(glcm, 'energy')[0, 0])

    all_new_features = pd.DataFrame({**new_cols, **glcm_features})
    df = pd.concat([df.reset_index(drop=True), all_new_features.reset_index(drop=True)], axis=1)
    
    return df

