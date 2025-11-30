model_config = {
    'vocab_size': 2871,                # PAD + CLS + 2869 codes
    'hidden_size': 288,
    'seg_vocab_size': 34,              # 34 possible visits
    'age_vocab_size': 2,               # age disabled
    'max_position_embedding': 1500,    # enough tokens
    'hidden_dropout_prob': 0.1,
    'num_hidden_layers': 6,
    'num_attention_heads': 12,
    'attention_probs_dropout_prob': 0.1,
    'intermediate_size': 512,
    'hidden_act': 'gelu',
    'initializer_range': 0.02,
}

feature_dict = {
    'word':True,
    'seg':True,
    'age':False,
    'position': True
}
