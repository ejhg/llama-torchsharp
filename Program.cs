using LLAMA;
using LLAMA.tokenizer.llama_3;
using TorchSharp;

var weightsDir = args[0];
var device = "cpu";

torch.manual_seed (100);

Console.WriteLine ("start");

ITokenizer tokenizer = File.Exists ($"{weightsDir}/tokenizer.model")
    ? new Llama3Tokenizer (
        $"{weightsDir}/tokenizer.model")
    : new Llama2Tokenizer (
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
