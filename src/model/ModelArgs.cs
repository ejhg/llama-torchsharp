using System.Text.Json.Serialization;
using TorchSharp;

namespace LLAMA;

public class ModelArgs
{
    [JsonPropertyName ("dim")] public int Dim { get; set; } = 4096;

    [JsonPropertyName ("n_layers")] public int NLayers { get; set; } = 32;

    [JsonPropertyName ("n_heads")] public int NHeads { get; set; } = 32;

    [JsonPropertyName ("n_kv_heads")] public int? NKVHeads { get; set; }

    [JsonPropertyName ("vocab_size")] public int VocabSize { get; set; } = -1;

    [JsonPropertyName ("multiple_of")] public int MultipleOf { get; set; } = 256;

    [JsonPropertyName ("ffn_dim_multiplier")]
    public float? FFNDimMultiplier { get; set; }

    [JsonPropertyName ("norm_eps")] public float NormEps { get; set; } = 1e-5f;

    [JsonPropertyName ("max_batch_size")] public int MaxBatchSize { get; set; } = 3;

    [JsonPropertyName ("max_seq_len")] public int MaxSeqLen { get; set; } = 1024;

    public torch.ScalarType Dtype => torch.ScalarType.BFloat16;
}
