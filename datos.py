import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_csv("hf://datasets/Tobi-Bueck/customer-support-tickets/dataset-tickets-multi-lang-4-20k.csv")

print(df.head())
print(df.columns)
print(df.info())

df.to_csv("customer-support-tickets.csv", index=False)

df = pd.read_csv("customer-support-tickets.csv")
print(df.head())
print(df.columns)
print(df.info())

df_en = df[df['language'] == 'en']

df_en.to_csv("customer-support-tickets-en.csv", index=False)

df_en = pd.read_csv("customer-support-tickets-en.csv")
print(df_en.head())
print(df_en.columns)
print(df_en.info())

df_limpio = df_en.dropna(subset=['subject', 'body', 'answer', 'type'])

df_limpio.to_csv("customer-support-tickets-en-limpio.csv", index=False)

df_limpio = pd.read_csv("customer-support-tickets-en-limpio.csv")
print(df_limpio.head())
print(df_limpio.columns)
print(df_limpio.info())

df_limpio['tags'] = df_limpio[['tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5', 'tag_6', 'tag_7', 'tag_8']].apply(
    lambda row: ','.join(row.dropna().astype(str)), axis=1
)

df_limpio = df_limpio.drop(columns=['tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5', 'tag_6', 'tag_7', 'tag_8'])

df_limpio.to_csv("customer-support-tickets-en-limpio-sin-tags.csv", index=False)

print(df_limpio.head())
print(df_limpio.columns)
print(df_limpio.info())

df_derfinitivo = df_limpio

df_derfinitivo['problem_text'] = df_derfinitivo['subject'] + ' ' + df_derfinitivo['body']

df_derfinitivo.to_csv("customer-support-tickets-en-derfinitivo.csv", index=False)
print(df_limpio.head())
print(df_limpio.columns)
print(df_limpio.info())