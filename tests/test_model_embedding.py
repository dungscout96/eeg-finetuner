from eeg_finetuner.foundation_model import get_foundation_model

def test_labram_embedding():
    model = get_foundation_model(
        model_name="labram",
        n_times=128,
        n_chans=64,
        n_outputs=0,
        emb_size=128
    )
    import torch
    dummy_input = torch.randn(4, 64, 128)  # batch_size=4, num_channels=64, input_size=128
    embeddings = model(dummy_input)
    assert embeddings.shape == (4, 128)  # batch_size=4, embedding_size=128