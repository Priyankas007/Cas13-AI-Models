# prediction function
def encodeRNA(sequences):
    encoded_rna = []
    for char in sequences:
        if(char == 'A'):
            encoded_rna.append([1.0,0.0,0.0,0.0,0.0,0.0])
        elif(char == 'C'):
            encoded_rna.append([0.0,1.0,0.0,0.0,0.0,0.0])
        elif(char == 'G'):
            encoded_rna.append([0.0,0.0,1.0,0.0,0.0,0.0])
        elif(char == 'U'):
            encoded_rna.append([0.0,0.0,0.0,1.0,0.0,0.0])
        elif(char == 'T'):
            encoded_rna.append([0.0,0.0,0.0,0.0,1.0,0.0])
        else:
            encoded_rna.append([0.0,0.0,0.0,0,0.0,1.0])
    return encoded_rna

def prepare_rna(sequences):
    encoded_rna = []
    for i in range (len(sequences)):
        encoded_rna.append(encodeRNA(sequences[i]))
    return encoded_rna
