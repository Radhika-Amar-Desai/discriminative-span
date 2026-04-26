MARKDOWN_RESULTS = """

### 1. Pneumonia CXR

#### 1.1 Downstream Performance

| Model           | Train Acc | Train F1 | Val Acc | Val F1 Score | Test Acc | Test F1 |
| ResNet18       | 0.9846    | 0.9848   | 0.6875  | 0.7619       | 0.4952   | 0.3811  |
| MobileNetV2    | 0.9884    | 0.9885   | 0.5000  | 0.6667       | 0.5192   | 0.3976  |
| EfficientNetB0 | 0.9884    | 0.9885   | 0.6875  | 0.6667       | 0.5192   | 0.3827  |



#### 1.2 Diagnostic Metrics (Embedding Space Analysis)

| Embedding Model   | Solver        |   Effective Rank |   Relative Error |   Explained Fraction |   Number of Pairs |   Dim |
| resnet18          | Least Squares |              182 |           0.754  |               0.246  |               259 |   512 |
| resnet18          | Ridge         |              182 |           0.8389 |               0.1611 |               259 |   512 |
| resnet18          | NNLS          |              182 |           0.9756 |               0.0244 |               259 |   512 |
| resnet18          | L1            |              182 |           0.9089 |               0.0911 |               259 |   512 |
| clip              | Least Squares |              145 |           0.7341 |               0.2659 |               259 |   512 |
| clip              | Ridge         |              145 |           0.8966 |               0.1034 |               259 |   512 |
| clip              | NNLS          |              145 |           0.9903 |               0.0097 |               259 |   512 |
| clip              | L1            |              145 |           0.9622 |               0.0378 |               259 |   512 |
| dinov2            | Least Squares |              151 |           0.5943 |               0.4057 |               259 |   384 |
| dinov2            | Ridge         |              151 |           0.8221 |               0.1779 |               259 |   384 |
| dinov2            | NNLS          |              151 |           0.9879 |               0.0121 |               259 |   384 |
| dinov2            | L1            |              151 |           0.8939 |               0.1061 |               259 |   384 |



### 2. Skin Lesion Dataset

#### 2.1 Downstream Performance

| Model           | Train Acc | Train F1 | Val Acc | Val F1 | Test Acc | Test F1 |
| ResNet18       | 0.8097    | 0.7705   | 0.9500  | 0.9524 | 0.8605   | 0.8537  |
| MobileNetV2    | 0.9972    | 0.9972   | 1.0000  | 1.0000 | 0.9302   | 0.9231  |
| EfficientNetB0 | 0.9503    | 0.9477   | 1.0000  | 1.0000 | 0.9419   | 0.9412  |



#### 2.2 Diagnostic Metrics (Embedding Space Analysis)

| Embedding Model   | Solver        |   Effective Rank |   Relative Error |   Explained Fraction |   Number of Pairs |   Dim |
| resnet18          | Least Squares |              203 |         0.34136  |               0.6586 |               352 |   512 |
| resnet18          | Ridge         |              203 |         0.463288 |               0.5367 |               352 |   512 |
| resnet18          | NNLS          |              203 |         0.687487 |               0.3125 |               352 |   512 |
| resnet18          | L1            |              203 |         0.783373 |               0.2166 |               352 |   512 |
| clip              | Least Squares |              200 |         0.206622 |               0.7934 |               352 |   512 |
| clip              | Ridge         |              200 |         0.319543 |               0.6805 |               352 |   512 |
| clip              | NNLS          |              200 |         0.557877 |               0.4421 |               352 |   512 |
| clip              | L1            |              200 |         0.684727 |               0.3153 |               352 |   512 |
| dinov2            | Least Squares |              190 |         0.121814 |               0.8782 |               352 |   384 |
| dinov2            | Ridge         |              190 |         0.356713 |               0.6433 |               352 |   384 |
| dinov2            | NNLS          |              190 |         0.64572  |               0.3543 |               352 |   384 |
| dinov2            | L1            |              190 |         0.699646 |               0.3004 |               352 |   384 |


### 3. Toy Watermark Dataset

#### 3.1 Downstream Performance

| Model           | Train Acc | Train F1 | Val Acc | Val F1 | Test Acc | Test F1 |
| ResNet18       | 0.9309    | 0.9354   | 1.0000  | 1.0000 | 0.9556   | 0.9615  |
| MobileNetV2    | 0.9814    | 0.9818   | 1.0000  | 1.0000 | 1.0000   | 1.0000  |
| EfficientNetB0 | 0.9825    | 0.9828   | 1.0000  | 1.0000 | 1.0000   | 1.0000  | 



#### 3.2 Diagnostic Metrics (Embedding Space Analysis)

| Embedding Model   | Solver        |   Effective Rank |   Relative Error |   Explained Fraction |   Number of Pairs |   Dim |
| resnet18          | Least Squares |              288 |        0.102834  |               0.8972 |               485 |   512 |
| resnet18          | Ridge         |              288 |        0.356454  |               0.6435 |               485 |   512 |
| resnet18          | NNLS          |              288 |        0.711918  |               0.2881 |               485 |   512 |
| resnet18          | L1            |              288 |        0.726766  |               0.2732 |               485 |   512 |
| clip              | Least Squares |              263 |        0.0817881 |               0.9182 |               485 |   512 |
| clip              | Ridge         |              263 |        0.262815  |               0.7372 |               485 |   512 |
| clip              | NNLS          |              263 |        0.473162  |               0.5268 |               485 |   512 |
| clip              | L1            |              263 |        0.5661    |               0.4339 |               485 |   512 |
| dinov2            | Least Squares |              250 |        2.23e09  |               1      |               485 |   384 |
| dinov2            | Ridge         |              250 |        0.31784   |               0.6822 |               485 |   384 |
| dinov2            | NNLS          |              250 |        0.547462  |               0.4525 |               485 |   384 |
| dinov2            | L1            |              250 |        0.640865  |               0.3591 |               485 |   384 |



### 4. Horses and Zebra

#### 4.1 Downstream Performance

| Model           | Train Acc | Train F1 | Test Acc | Test F1 |
| ResNet18       | 0.9091    | 0.9137   | 0.8375   | 0.8660  |
| MobileNetV2    | 0.9996    | 0.9996   | 0.8792   | 0.8835  |
| EfficientNetB0 | 0.9763    | 0.9757   | 0.9292   | 0.9328  | 



#### 4.2 Diagnostic Metrics (Embedding Space Analysis)

| Embedding Model   | Solver        |   Effective Rank |   Relative Error |   Explained Fraction |   Number of Pairs |   Dim |
| resnet18          | Least Squares |              343 |         2.06e09 |               1      |              1117 |   512 |
| resnet18          | Ridge         |              343 |         0.417489 |               0.5825 |              1117 |   512 |
| resnet18          | NNLS          |              343 |         0.983051 |               0.0169 |              1117 |   512 |
| clip              | Least Squares |              307 |         1.68e09 |               1      |              1117 |   512 |
| clip              | Ridge         |              307 |         0.380944 |               0.6191 |              1117 |   512 |
| clip              | NNLS          |              307 |         0.946506 |               0.0535 |              1117 |   512 |
| dinov2            | Least Squares |              283 |         9.55e10 |               1      |              1117 |   384 |
| dinov2            | Ridge         |              283 |         0.329001 |               0.671  |              1117 |   384 |
| dinov2            | NNLS          |              283 |         0.940155 |               0.0598 |              1117 |   384 |



### 5. Apples and Oranges

#### 5.1 Downstream Performance

| Model           | Train Acc | Train F1 | Test Acc | Test F1 | Val Acc | Val F1 |
| ResNet18       | 0.9151    | 0.9112   | 0.8785   | 0.8649  | 1.0000  | 1.0000 |
| MobileNetV2    | 0.9814    | 0.9812   | 0.8947   | 0.8850  | 1.0000  | 1.0000 |
| EfficientNetB0 | 0.9839    | 0.9840   | 0.9291   | 0.9251  | 1.0000  | 1.0000 |



#### 5.2 Diagnostic Metrics (Embedding Space Analysis)


| Embedding Model   | Solver        |   Effective Rank |   Relative Error |   Explained Fraction |   Number of Pairs |   Dim |
| resnet18          | Least Squares |              351 |         1.21e09 |               1      |               995 |   512 |
| resnet18          | Ridge         |              351 |         0.375139 |               0.6249 |               995 |   512 |
| resnet18          | NNLS          |              351 |         0.679303 |               0.3207 |               995 |   512 |
| resnet18          | L1            |              351 |         0.663738 |               0.3363 |               995 |   512 |
| clip              | Least Squares |              329 |         7.98e10 |               1      |               995 |   512 |
| clip              | Ridge         |              329 |         0.290121 |               0.7099 |               995 |   512 |
| clip              | NNLS          |              329 |         0.686178 |               0.3138 |               995 |   512 |
| clip              | L1            |              329 |         0.695665 |               0.3043 |               995 |   512 |
| dinov2            | Least Squares |              268 |         9.05e10 |               1      |               995 |   384 |
| dinov2            | Ridge         |              268 |         0.413526 |               0.5865 |               995 |   384 |
| dinov2            | NNLS          |              268 |         0.790725 |               0.2093 |               995 |   384 |
| dinov2            | L1            |              268 |         0.693928 |               0.3061 |               995 |   384 |



### 6. TB Dataset

#### 6.1 Downstream Performance

| Model           | Train Acc | Train F1 | Val Acc | Val F1 | Test Acc | Test F1 |
| ResNet18       | 0.8874    | 0.8731   | 0.7500  | 0.6667 | 0.5000   | 0.0000  |
| MobileNetV2    | 0.9835    | 0.9833   | 0.5200  | 0.2727 | 0.5000   | 0.0196  |
| EfficientNetB0 | 0.9588    | 0.9600   | 0.8600  | 0.8478 | 0.5100   | 0.0755  |

#### 6.2 Diagnostic Metrics (Embedding Space Analysis)

| Embedding Model   | Solver        |   Effective Rank |   Relative Error |   Explained Fraction |   Number of Pairs |   Dim |
| resnet18          | Least Squares |              135 |         0.758705 |               0.2413 |               182 |   512 |
| resnet18          | Ridge         |              135 |         0.810364 |               0.1896 |               182 |   512 |
| resnet18          | NNLS          |              135 |         0.984289 |               0.0157 |               182 |   512 |
| resnet18          | L1            |              135 |         0.910664 |               0.0893 |               182 |   512 |
| clip              | Least Squares |              113 |         0.624809 |               0.3752 |               182 |   512 |
| clip              | Ridge         |              113 |         0.678269 |               0.3217 |               182 |   512 |
| clip              | NNLS          |              113 |         0.929987 |               0.07   |               182 |   512 |
| clip              | L1            |              113 |         0.866587 |               0.1334 |               182 |   512 |
| dinov2            | Least Squares |              119 |         0.661714 |               0.3383 |               182 |   384 |
| dinov2            | Ridge         |              119 |         0.786722 |               0.2133 |               182 |   384 |
| dinov2            | NNLS          |              119 |         0.959143 |               0.0409 |               182 |   384 |
| dinov2            | L1            |              119 |         0.88635  |               0.1136 |               182 |   384 |
"""