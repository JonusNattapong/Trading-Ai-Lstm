"""
Simple forward pass test for LSTM models (simple, bidirectional, attention)
Runs `forward` for each model with dummy inputs and validates output shape.
"""

try:
    import torch
    from lstm_model import create_model
    import config
except Exception as e:
    print('Torch import failed:', e)
    print('Please install PyTorch (CPU-only recommended on Windows):')
    print('  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu')
    raise


def test_forward_simple(input_size=50, seq_len=60, batch_size=2):
    model = create_model('simple', input_size)
    model.eval()
    x = torch.randn(batch_size, seq_len, input_size)
    with torch.no_grad():
        out = model(x)
    print('Simple LSTM output shape:', out.shape)
    return out.shape


def test_forward_bidirectional(input_size=50, seq_len=60, batch_size=2):
    model = create_model('bidirectional', input_size)
    model.eval()
    x = torch.randn(batch_size, seq_len, input_size)
    with torch.no_grad():
        out = model(x)
    print('Bidirectional LSTM output shape:', out.shape)
    return out.shape


def test_forward_attention(input_size=50, seq_len=60, batch_size=2):
    model = create_model('attention', input_size)
    model.eval()
    x = torch.randn(batch_size, seq_len, input_size)
    with torch.no_grad():
        out = model(x)
    print('Attention LSTM output shape:', out.shape)
    return out.shape


if __name__ == '__main__':
    print('Running forward pass tests')
    input_size = 50
    seq_len = config.SEQUENCE_LENGTH if hasattr(config, 'SEQUENCE_LENGTH') else 60

    try:
        s_out = test_forward_simple(input_size, seq_len)
        b_out = test_forward_bidirectional(input_size, seq_len)
        a_out = test_forward_attention(input_size, seq_len)

        if s_out[0] == 2 and b_out[0] == 2 and a_out[0] == 2:
            print('\nAll forward pass tests succeeded!')
        else:
            print('\nUnexpected output shapes:', s_out, b_out, a_out)

    except Exception as exc:
        print('Forward pass tests failed:', exc)