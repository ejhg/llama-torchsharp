namespace LLAMA;

public class ConfigurationParamsJson
{
    public int dim { get; set; } = 4096;

    public int n_layers { get; set; } = 32;

    public int n_heads { get; set; } = 32;

    public float norm_eps { get; set; } = 1e-5f;

    public int multiple_of { get; set; } = 256;

    public int? n_kv_heads { get; set; }

    public float? ffn_dim_multiplier { get; set; }
}