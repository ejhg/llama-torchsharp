using TorchSharp;

namespace LLAMA;

public class ConfigurationParams
{
    public int dim { get; set; } = 4096;

    public int n_layers { get; set; } = 32;

    public int n_heads { get; set; } = 32;

    public int? n_kv_heads { get; set; }

    public int vocab_size { get; set; } = -1;

    public int multiple_of { get; set; } = 256;

    public float? ffn_dim_multiplier { get; set; }

    public float norm_eps { get; set; } = 1e-5f;

    public int max_batch_size { get; set; } = 3;

    public int max_seq_len { get; set; } = 1024;

    public torch.ScalarType Dtype => torch.ScalarType.BFloat16;
}
