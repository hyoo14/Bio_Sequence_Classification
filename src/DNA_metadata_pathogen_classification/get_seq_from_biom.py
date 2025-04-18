# !pip install biom-format



from biom import load_table
table = load_table("/content/reference-hit.biom")
print(table.shape)  # (OTUs, samples)
print(table.ids(axis='sample'))  # sample IDs
print(table.ids(axis='observation'))  # OTU IDs







