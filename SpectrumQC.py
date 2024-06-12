import streamlit as st
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
import numpy as np

import pathlib
import os
import struct
from typing import List, Tuple

@st.cache_resource
def load_model():
    with open('EASTER_QC_v3.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to make predictions
def transform(model, data):
    values = model_one.transform(data)
    return values



class SpaFile:
    def __init__(self, spa_path: str = ""):
        self.file_path = spa_path
        self.file_name = os.path.basename(spa_path) if spa_path else ""

    def load_spectrum(self) -> Tuple[List[float], List[float]]:
        return self._load_spectrum(self.file_path)

    def _load_spectrum(self, fn: str) -> Tuple[List[float], List[float]]:
        intensities = []
        spectrum = []

        intensity_start_offset = 1852
        max_wavenum_offset = 1600
        total_num_vals_offset = 1588
        total_num_vals = 7468  # Default value for resolution of 4 cm^-1
        max_wavenum = 0.0
        min_wavenum = 0.0
        wavenum_step = 0.0

        if os.path.exists(fn):
            # Read total number of intensity entries (list length) from offset 1588
            with open(fn, 'rb') as reader:
                reader.seek(total_num_vals_offset)
                total_num_vals = struct.unpack('H', reader.read(2))[0]

            # Read minimum and maximum wavenumbers (wavenumber range) from offset 1600
            with open(fn, 'rb') as reader:
                reader.seek(max_wavenum_offset)
                max_wavenum = struct.unpack('f', reader.read(4))[0]
                min_wavenum = struct.unpack('f', reader.read(4))[0]

            # Read individual intensity values starting from offset 1852
            with open(fn, 'rb') as reader:
                reader.seek(intensity_start_offset)
                for _ in range(total_num_vals):
                    intensity = struct.unpack('f', reader.read(4))[0]
                    intensities.append(intensity)

            wavenum_step = (max_wavenum - min_wavenum) / len(intensities)

        current_wavenum = max_wavenum
        for intensity in intensities:
            spectrum.append((current_wavenum, intensity))
            current_wavenum -= wavenum_step

        # Separate the tuples into two lists: wavenumbers and spectra
        wavenumbers, spectra = zip(*spectrum) if spectrum else ([], [])

        return np.array(list(wavenumbers)), np.array(list(spectra))

    def __str__(self):
        return self.file_path




def read_spa(file):

    file.seek(564)
    Spectrum_Pts = np.frombuffer(file.read(4), np.int32)[0]
    
    file.seek(30)
    SpectraTitles = np.frombuffer(file.read(255), np.uint8)
    SpectraTitles = ''.join([chr(x) for x in SpectraTitles if x != 0])

    file.seek(576)
    Max_Wavenum = np.frombuffer(file.read(4), np.single)[0]
    Min_Wavenum = np.frombuffer(file.read(4), np.single)[0]
    
    Wavenumbers = np.flip(np.linspace(Min_Wavenum, Max_Wavenum, Spectrum_Pts))

    file.seek(288)
    
    Flag = 0
    while Flag != 3:
        Flag = np.frombuffer(file.read(2), np.uint16)[0]
    
    DataPosition = np.frombuffer(file.read(2), np.uint16)[0]
    file.seek(DataPosition)

    Spectra = np.frombuffer(file.read(Spectrum_Pts * 4), np.single)
    return Spectra, 1e7 / Wavenumbers, SpectraTitles


# Streamlit app
def main():
    st.title("Spectral Quality Check (EASTER)")
        
    
    uploaded_file = st.file_uploader("Choose spectrum", type="spa")


    if uploaded_file is not None:

        

        Spectra, Wavelengths, SpectraTitles = read_spa(uploaded_file)

        

        if len(Spectra) == 0:

            uploaded_file2 = st.file_uploader("Choose same spectrum", type="spa")

            if uploaded_file2 is not None:
                temp_file_path = os.path.join("temp.spa")
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file2.read())
        

                spa = SpaFile(temp_file_path)

                waves, spectra = spa.load_spectrum()

                #st.write(len(spectra))


                X = spectra

                X_scaled0 = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

                model = load_model()
                value0 = transform(model, X_scaled0.reshape((1,len(spectra))))

                
                # Display predictions
                st.write("Spectral QC outcome:")
                if value0.T[0] > 5:
                    st.write('Email Michael')
                else:
                    st.write('Good quality spectrum')

        else:

            #mm2 = minmax_scale()

            X1 = Spectra

            X_scaled1 = (X1 - X1.min(axis=0)) / (X1.max(axis=0) - X1.min(axis=0))
            

           
            #st.write('This one') 

            model1 = load_model()
            value1 = transform(model1, X_scaled1.reshape((1,len(Spectra))))

                            
            # Display predictions
            st.write("Spectral QC outcome:")
            if value1.T[0] > 5:
                st.write('Email Michael')
            else:
                st.write('Goog quality spectrum')
        

if __name__ == "__main__":
    main()
