class CFG:
    BERT_PATH = "microsoft/codebert-base"
    SAVE_PATH = "./output"
    LOAD_DATA_PATH = "./data/"
    MAX_LEN = 64
    TOTAL_MAX_LEN = 512
    ACCUMULATION_STEPS = 4
    LR = 5e-5
    
    BS = 48 #batch size
    NW = 2 # num woker
    EPOCHS = 3 
    DATA_DIR = Path('../content/AI4Code')