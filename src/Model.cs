using System.Diagnostics;
using System.Text.Json;
using TorchSharp;
using TorchSharp.PyBridge;

namespace LLAMA;

static class Model
{
    public static Transformer build (
        string modelFolder,
        ITokenizer tokenizer,
        int maxSeqLen,
        int maxBatchSize,
        string paramJsonPath = "params.json",
        string modelWeightPath = "consolidated.00.pth",
        string device = "cpu"
    ) {
        var stopWatch = new Stopwatch ();
        stopWatch.Start ();

        paramJsonPath = Path.Combine (modelFolder, paramJsonPath);
        var modelArgs = JsonSerializer.Deserialize<ConfigurationParams> (File.ReadAllText (paramJsonPath));

        modelArgs.vocab_size = tokenizer.VocabSize;
        modelArgs.max_seq_len = maxSeqLen;
        modelArgs.max_batch_size = maxBatchSize;

        torch.set_default_dtype (torch.bfloat16);

        // print model args
        var modelArgsJson = JsonSerializer.Serialize (modelArgs, new JsonSerializerOptions { WriteIndented = true });
        Console.WriteLine ($"modelArgs: {modelArgsJson}");

        var model = new Transformer (modelArgs);
        var loadedParameters = new Dictionary<string, bool> ();

        Console.WriteLine ("loading checkpoint");
        model.load_py (
            location: Path.Combine (modelFolder, modelWeightPath),
            strict: false,
            loadedParameters: loadedParameters);

        // print loaded parameters
        foreach (var (key, value) in loadedParameters.OrderBy (x => x.Key)) {
            Console.WriteLine ($"loadedParameters: {key} {value}");
        }

        model = model.to (device);

        stopWatch.Stop ();
        Console.WriteLine ($"Loading checkpoint took {stopWatch.ElapsedMilliseconds} ms");

        return model;
    }
}
