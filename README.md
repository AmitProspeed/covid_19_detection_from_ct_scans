# covid_19_detection_from_ct_scans

## Commands to run code

pip3 install tensorflow sklearn Pillow numpy scipy opencv-contrib-python

python3 main.py

For testing: provide relative image path in the terminal when asked
Eg: 'dataset/CT_NonCOVID/0.jpg'

## Original Paper Results

![Arch_Image](https://github.com/AmitProspeed/covid_19_detection_from_ct_scans/blob/main/OriginalResults.png)

## Server Results

![Arch_Image](https://github.com/AmitProspeed/covid_19_detection_from_ct_scans/blob/main/server_result.png)
## Results:

### DenseNet(Best Result)

Accuracy: **0.8954414414414416** + 0.017800133050865665  
Area Under Curve: **0.9451266968325791** + 0.01732083766849508  
Recall: **0.9055462184873949** + 0.04035573753343885  
Presicion: **0.8790384397701472** + 0.04465206377630674  
f1 score: **0.8903512693210945** + 0.016475958527555734  

### InceptionV3

Accuracy: 0.7923243243243243 + 0.03397103183562358  
Area Under Curve: 0.8551754471019176 + 0.054920764052541274  
Recall: 0.7278151260504201 + 0.0559141881898442  
Presicion: 0.8125817618031188 + 0.05110884278597921  
f1 score: 0.7658766330847202 + 0.037378742239523324  
### ResNet50V2

Accuracy: 0.7172072072072073 + 0.04480601444317083  
Area Under Curve: 0.836914188752424 + 0.029827148493351663  
Recall: 0.5736974789915966 + 0.17743874082024483  
Presicion: 0.7978379583102222 + 0.09733178837672064  
f1 score: 0.6433099215360623 + 0.08579116761363623  

### ResNet50V1

Accuracy: 0.7307387387387387 + 0.059821784402777356  
Area Under Curve: 0.8158674315880198 + 0.04642950111493882  
Recall: 0.7052100840336134 + 0.09328927169888922  
Presicion: 0.7296806119304476 + 0.10214278257512288  
f1 score: 0.7098448213443521 + 0.06197074090652394  


### MobileNetV1

Accuracy: 0.8942342342342343 + 0.03472984073549991  
Area Under Curve: 0.9447558177117001 + 0.021410335161163856  
Recall: 0.8884033613445379 + 0.05320701563000841  
Presicion: 0.8882176897136329 + 0.04578980032540859  
f1 score: 0.8868963388886888 + 0.03662779986147132  

### MobileNetV2

Accuracy: 0.857981981981982 + 0.024423926872629547  
Area Under Curve: 0.9276141995259641 + 0.0274733957058049  
Recall: 0.8825210084033615 + 0.039267712587939076  
Presicion: 0.8284741059131303 + 0.039821328984618126  
f1 score: 0.8533989375325175 + 0.02370271903395987  

## Code Screenshot

![Arch_Image](https://github.com/AmitProspeed/covid_19_detection_from_ct_scans/blob/main/code.png)


## Reference
A Novel and Reliable Deep Learning Web-Based Tool to Detect COVID-19 Infection from Chest CT-Scan - https://arxiv.org/pdf/2006.14419.pdf

