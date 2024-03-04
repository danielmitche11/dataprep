import dataprep

df = dataprep.import_data("sample_data.csv")
df = dataprep.clean_data(df)
dataprep.export_data(df, "clean_sample")
