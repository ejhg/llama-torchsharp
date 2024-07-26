using LLAMA;
using TorchSharp;

var weightsDir = args[0];
var device = "cpu";

torch.manual_seed (100);

Console.WriteLine ("start");

var tokenizer = new BPETokenizer (
    "resources/vocab.json",
    "resources/merges.txt");

var model = Model.build (
    modelFolder: weightsDir,
    tokenizer: tokenizer,
    maxSeqLen: 128,
    maxBatchSize: 1,
    device: device);

var prompts = new[] {
    "I believe the meaning of life is",
};

var result = Inference.TextCompletion (
    model,
    tokenizer,
    prompts,
    temperature: 0,
    echo: true,
    device: device);

foreach (var item in result) {
    Console.WriteLine ($"generation: {item.generation}");
}
