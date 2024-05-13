from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline

ordinal_encode_pf = Pipeline(
    steps=[(
        'encode', OrdinalEncoder(categories=[sorted({'I': 0, 'S': 1, 'M': 2, 'B': 3, 'W': 4}, key={'I': 0, 'S': 1, 'M': 2, 'B': 3, 'W': 4}.get)])
    )]
)

ordinal_encode_tar = Pipeline(
    steps=[(
        'encode', OrdinalEncoder(categories=[sorted({'High': 1, 'Low': 0}, key={'High': 1, 'Low': 0}.get)])
    )]
)

onehot_encode = Pipeline(
    steps=[(
        'encode', OneHotEncoder(sparse_output=False)
    )]
)

ordinal_encode = Pipeline(
    steps=[(
        'encode', OrdinalEncoder()
    )]
)