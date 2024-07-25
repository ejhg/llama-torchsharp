using LLAMA;
using TorchSharp;

Console.WriteLine ("start");

var vocabPath = @"vocab.json";
var mergesPath = @"merges.txt";
var tokenizer = new BPETokenizer (vocabPath, mergesPath);

var checkpointDirectory = args[0];

var device = "cpu";

torch.manual_seed (100);
var model = Model.build (
    modelFolder: checkpointDirectory,
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
