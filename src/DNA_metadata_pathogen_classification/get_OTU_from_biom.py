
# !pip install biom-format


from biom import load_table
table = load_table("/content/otu_table.biom")


print(table.shape)  # (OTUs, samples)

print(table.ids(axis='sample'))  # sample IDs

print(table.ids(axis='observation'))  # OTU IDs


# get abundance per specific sample ID 
abundance = table.data('10442.SSB780', axis='sample')
