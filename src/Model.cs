using System.Diagnostics;
using System.Text.Json;
using TorchSharp;
using llama.unpickler;

namespace LLAMA;

static class Model
{
    public static Transformer build (
        string modelFolder,
        ITokenizer tokenizer,
        int maxSeqLen,
        int maxBatchSize,
        string device,
        string paramJsonPath = "params.json",
        string modelWeightPath = "consolidated.00.pth"
    ) {
        var stopWatch = new Stopwatch ();
        stopWatch.Start ();

        paramJsonPath = Path.Combine (modelFolder, paramJsonPath);
        var json = JsonSerializer.Deserialize<ConfigurationParamsJson> (File.ReadAllText (paramJsonPath));

        var modelArgs = new ConfigurationParams {
            dim = json.dim,
            n_layers = json.n_layers,
            n_heads = json.n_heads,
            norm_eps = json.norm_eps,
            multiple_of = json.multiple_of,
            n_kv_heads = json.n_kv_heads ?? json.n_heads,
            rope_theta = json.rope_theta ?? 500000,
            ffn_dim_multiplier = json.ffn_dim_multiplier,
            vocab_size = tokenizer.VocabSize,
            max_seq_len = maxSeqLen,
            max_batch_size = maxBatchSize,
            use_scaled_rope = json.use_scaled_rope ?? false,
            Dtype = device == "mps"
                ? torch.ScalarType.Float16
                : torch.ScalarType.BFloat16,
        };

        torch.set_default_dtype (modelArgs.Dtype);

        // print model args
        var modelArgsJson = JsonSerializer.Serialize (modelArgs, new JsonSerializerOptions { WriteIndented = true });
        Console.WriteLine ($"modelArgs: {modelArgsJson}");

        var model = new Transformer (modelArgs);

        Console.WriteLine ("loading checkpoint");
        model.load_weights (
            location: Path.Combine (modelFolder, modelWeightPath));

        model = model.to (device);

        stopWatch.Stop ();
        Console.WriteLine ($"Loading checkpoint took {stopWatch.ElapsedMilliseconds} ms");

        return model;
    }
}
