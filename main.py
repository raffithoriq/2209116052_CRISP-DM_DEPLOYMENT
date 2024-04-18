import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pickle
import seaborn as sns
from sklearn.cluster import KMeans
import mlxtend
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_csv('amazon.csv')

# NAVBAR
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", ['Home', 'EDA', 'MODELING'])

if selection == 'Home':
    st.title("Amazon UK Grocery Dataset")

    # Tambahkan URL gambar
    image_url = 'https://storage.googleapis.com/kaggle-datasets-images/3658758/6352802/8638fc8bfa162e9f617cb9fbdf40478c/dataset-cover.jpeg?t=2023-08-23-14-52-17'

    # Tampilkan gambar menggunakan st.image()
    st.image(image_url, use_column_width=True)

    st.title("Business objective")
    st.write("Amazon UK Grocery berusaha untuk meningkatkan retensi pelanggan dengan menawarkan opsi berlangganan produk-produk segar. Melalui program Fresh Picks, mereka bertujuan menciptakan kebiasaan belanja yang konsisten dan berkelanjutan, yang dapat meningkatkan loyalitas pelanggan. Dengan memiliki pelanggan berlangganan, Amazon UK Grocery juga dapat memperkirakan pendapatan yang lebih konsisten dari program tersebut, karena pelanggan akan melakukan pembelian secara teratur setiap periode pengiriman. Selain itu, program Fresh Picks juga membuka peluang untuk meningkatkan keterlibatan pelanggan. Amazon UK Grocery dapat berkomunikasi secara reguler dengan pelanggan berlangganan tentang pilihan produk, memberikan rekomendasi, dan menawarkan penawaran khusus. Hal ini dapat memperkuat hubungan dengan pelanggan dan menjaga mereka terlibat secara aktif dalam belanja di platform tersebut. Dari sisi citra merek, fokus Amazon UK Grocery pada produk-produk segar dan kualitas dapat membantu membangun citra merek yang sehat. Mereka dapat dipandang sebagai destinasi belanja yang dapat dipercaya untuk kebutuhan makanan segar dan sehat, yang pada gilirannya dapat meningkatkan kepercayaan pelanggan dan loyalitas mereka terhadap merek tersebut. Terakhir, dengan menyediakan opsi berlangganan untuk produk-produk segar, Amazon UK Grocery dapat mendorong pelanggan untuk menambahkan produk tambahan ke pesanan mereka. Hal ini dapat meningkatkan nilai pesanan rata-rata dan pendapatan keseluruhan perusahaan, memberikan dampak positif secara finansial.")
    

    st.title("Assess Situation")
    st.write("Untuk menjadikan model bisnis berlangganan Fresh Picks Amazon UK Grocery sebagai strategi yang berkelanjutan, beberapa langkah penting harus diambil. Pertama, perusahaan harus terus melakukan analisis dan penyesuaian berkelanjutan berdasarkan umpan balik pelanggan dan perubahan pasar. Ini bertujuan untuk menjaga program Fresh Picks tetap relevan dan menarik dalam jangka panjang. Selanjutnya, kelayakan finansial dari program ini perlu dipertimbangkan dengan melakukan perhitungan biaya operasional dan potensi pendapatan jangka panjang. Selain itu, kesiapan pasar harus dievaluasi untuk memastikan bahwa konsumen di pasar Inggris sudah siap untuk menggunakan layanan berlangganan produk segar secara teratur. Infrastruktur logistik juga perlu dipertimbangkan agar dapat mendukung pengiriman produk segar tepat waktu dan dalam kondisi yang baik kepada pelanggan. Analisis persaingan diperlukan untuk memahami bagaimana Fresh Picks akan bersaing dan berbeda di pasar makanan segar dan berlangganan. Terakhir, analisis pasar dan permintaan konsumen diperlukan untuk memahami kebutuhan dan minat konsumen terhadap produk segar serta layanan pengiriman secara teratur. Dengan menggabungkan semua langkah ini dalam strategi keseluruhan, Amazon UK Grocery dapat memastikan bahwa model bisnis berlangganan Fresh Picks dapat berkelanjutan dalam jangka panjang sambil memberikan nilai tambah yang signifikan bagi perusahaan dan pelanggan mereka.")
    

    st.title("Data Mining Goals")
    st.write("Amazon UK Grocery telah mengadopsi strategi berbasis data yang komprehensif untuk meningkatkan pengalaman pelanggan dan efisiensi operasional dalam program Fresh Picks. Mereka menggunakan data mining untuk mengidentifikasi faktor-faktor yang mempengaruhi retensi pelanggan, personalisasi rekomendasi produk, segmentasi pelanggan, prediksi permintaan, analisis kepuasan pelanggan, dan optimalisasi operasional. Dengan strategi ini, Amazon UK Grocery dapat mempertahankan pelanggan lebih lama, meningkatkan relevansi produk, mengoptimalkan stok persediaan, meningkatkan kepuasan pelanggan, dan mengurangi biaya operasional. Dengan demikian, mereka dapat meningkatkan kesuksesan dan kelangsungan program Fresh Picks sambil memberikan nilai tambah yang signifikan bagi pelanggan dan perusahaan secara keseluruhan.")

    st.title("Project Plan")
    st.write("Amazon UK Grocery memulai program Fresh Picks dengan langkah-langkah yang terstruktur dan terencana secara sistematis. Mereka memulai dengan perencanaan yang mencakup penetapan tujuan program, pembentukan tim proyek, alokasi sumber daya, dan penelitian pasar untuk memahami kebutuhan pelanggan. Selanjutnya, dalam tahap persiapan, mereka mengidentifikasi produk, menjalin kemitraan dengan pemasok, merancang platform, dan mengevaluasi infrastruktur logistik untuk memastikan ketersediaan dan pengiriman produk yang tepat waktu. Tahap implementasi melibatkan pembangunan platform, persiapan kampanye pemasaran, pelatihan staf, dan penyusunan prosedur pengiriman yang efisien. Saat program Fresh Picks diluncurkan, Amazon UK Grocery memonitor kinerja program, menanggapi umpan balik pelanggan, dan melakukan penyesuaian jika diperlukan. Terakhir, tahap pemeliharaan dan pengembangan melibatkan evaluasi kinerja program, inovasi produk dan layanan berdasarkan hasil evaluasi, serta pemantauan tren pasar untuk tetap relevan dan kompetitif. Dengan strategi yang terorganisir dan proaktif ini, Amazon UK Grocery dapat menjaga keberhasilan dan kelangsungan program Fresh Picks sambil terus memberikan nilai tambah yang signifikan bagi pelanggan dan perusahaan.")

elif selection == 'EDA':
    st.title('Amazon UK Grocery Dataset - EDA')

    # Display the first few rows of the dataset
    st.subheader('Dataset Preview:')
    st.write(df.head())

    # Display descriptive statistics
    st.subheader('Descriptive Statistics:')
    st.write(df.describe())

    # Display missing values
    st.subheader('Missing Values:')
    st.write(df.isnull().sum())


    st.write("### Distribution of Numerical Features")
    plt.figure(figsize=(12, 6))
    df.select_dtypes(include=['float64', 'int64']).hist(bins=20, color='skyblue', edgecolor='black', linewidth=1.5)
    plt.tight_layout()
    st.pyplot()

    st.subheader('Correlation Matrix')
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    st.pyplot()

elif selection == 'MODELING':
    st.title('MODELING')
    st.write('Welcome to the MODELING.')

    # Input nilai fitur untuk prediksi
    CustomerID = st.text_input('Customer ID')
    Price = st.text_input('Price')
    Quantity = st.text_input('Quantity')
    InvoiceDate = st.text_input('Invoice Date')

    # Model prediksi
    # Disini, kita tidak memiliki informasi target untuk melakukan pemodelan prediksi
    # Jadi, kita akan membuat sebuah contoh saja untuk menunjukkan bagaimana alur kerja pemodelan.

    if st.button('Predict'):
        # Cek apakah semua nilai input telah diisi
        if CustomerID and Price and Quantity and InvoiceDate:
            # Contoh model prediksi
            # Misalnya, kita membuat model prediksi sederhana dengan syarat tertentu
            if float(Price) > 13.499002587710912:
                prediction_result = "high"
            else:
                prediction_result = "low"

            # Tampilkan hasil prediksi
            st.write(f'Predicted: {prediction_result}')
        else:
            st.error("Please fill in all input values before predicting.")

# Tambahkan bagian untuk EDA dan rangkuman dataset di sini
    






