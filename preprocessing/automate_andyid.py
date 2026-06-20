import pandas as pd
import os

def preprocess_data(input_path='namadataset_raw/nama_file_dataset.csv', output_path='namadataset_preprocessing/processed_data.csv'):
    """
    Fungsi untuk melakukan preprocessing data secara otomatis.
    Mengambil data dari input_path, melakukan transformasi, dan menyimpan ke output_path.
    """
    print(f"Memulai preprocessing dari: {input_path}")
    try:
        df = pd.read_csv(input_path)
        print("Data berhasil dimuat.")

        # --- Preprocessing ---
        # 1. Handling missing values
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown', inplace=True)

        # 2. Hapus duplikat
        df.drop_duplicates(inplace=True)

        # 3. Encoding fitur kategorikal
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = pd.factorize(df[col])[0]
        
        # Pastikan direktori output ada
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # Contoh: Menyimpan hasil preprocessing
        df.to_csv(output_path, index=False)
        print(f"Preprocessing selesai. Data disimpan ke: {output_path}")
        return output_path
    except FileNotFoundError:
        print(f"Error: File input tidak ditemukan di {input_path}. Pastikan file ada di 'namadataset_raw/'.")
        return None
    except Exception as e:
        print(f"Terjadi error saat preprocessing: {e}")
        return None

if __name__ == "__main__":
    # Contoh penggunaan: Panggil fungsi preprocess_data
    # Sesuaikan 'nama_file_dataset.csv' jika nama file Anda berbeda
    # Pastikan file dataset ada di folder 'namadataset_raw/'
    # Contoh pembuatan file dummy jika belum ada:
    raw_data_dir = 'namadataset_raw'
    output_data_dir = 'namadataset_preprocessing'
    dataset_filename = 'nama_file_dataset.csv' # Ganti dengan nama file Anda
    
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(output_data_dir, exist_ok=True)

    dummy_input_path = os.path.join(raw_data_dir, dataset_filename)
    if not os.path.exists(dummy_input_path):
         dummy_data = {'col1': [1, 2, 3], 'col2': ['A', 'B', 'C']}
         pd.DataFrame(dummy_data).to_csv(dummy_input_path, index=False)
         print(f"File dummy '{dataset_filename}' dibuat di '{raw_data_dir}/'.")

    preprocessed_file = preprocess_data(input_path=dummy_input_path, output_path=os.path.join(output_data_dir, 'processed_data.csv')) 
    
    if preprocessed_file:
        print(f"Script preprocessing berhasil dijalankan.")
    else:
        print("Script preprocessing gagal dijalankan.")
