using System.Diagnostics;
using TorchSharp;

namespace LLAMA;

public record CompletionPrediction (string generation, string[]? tokens, float[]? logProbs);

static class Inference
{
    static (int[][], float[][]?) Generate (
        Transformer transformer,
        ITokenizer tokenizer,
        int[][] promptTokens,
        int maxGenLen,
        float temperature,
        float topP,
        bool logProbs,
        bool echo,
        string device
    ) {
        torch.Tensor? tokenLogProbs = null;
        var batch = promptTokens.Length;
        var param = transformer.args;
        Debug.Assert (batch <= param.max_batch_size, "Batch size should be less than or equal to the max batch size");

        var minPromptLen = promptTokens.Min (x => x.Length);
        var maxPromptLen = promptTokens.Max (x => x.Length);
        Debug.Assert (maxPromptLen <= param.max_seq_len, "Prompt length should be less than or equal to the max sequence length");

        var totalLen = Math.Min (maxPromptLen + maxGenLen, param.max_seq_len);

        var tokens = torch.full (new long[] {
            batch,
            totalLen
        }, tokenizer.PadId, dtype: torch.int64, device: device);
        for (var i = 0; i < batch; i++) {
            var promptLen = promptTokens[i].Length;
            tokens[i, ..promptLen] = torch.tensor (promptTokens[i], dtype: torch.int64, device: device);
        }

        if (logProbs) {
            tokenLogProbs = torch.zeros (batch, totalLen, tokenizer.VocabSize, dtype: torch.float32, device: device);
        }

        using var no_grad = torch.no_grad ();
        using var inference_mode = torch.inference_mode ();

        var prevPos = 0;
        var eosReached = torch.tensor (new bool[batch], device: device);
        var inputTextMask = tokens != tokenizer.PadId;

        for (int curPos = minPromptLen; curPos < totalLen; curPos++) {
            using var scope = torch.NewDisposeScope ();

            Console.WriteLine ("tensors: " + torch.Tensor.TotalCount + "/" + torch.Tensor.PeakCount);

            var stopWatch = new Stopwatch ();
            stopWatch.Start ();

            var logits = transformer.forward (tokens[.., prevPos..curPos], prevPos);

            var nextToken = temperature > 0
                ? SampleTopP (logits, topP, temperature)
                : torch.argmax (logits[.., -1], dim: -1);

            nextToken = nextToken.reshape (-1);
            // # only replace token if prompt has already been generated
            nextToken = torch.where (inputTextMask[.., curPos], tokens[.., curPos], nextToken);

            // print nextToken
            tokens[.., curPos] = nextToken;
            if (logProbs) {
                tokenLogProbs![.., (prevPos + 1) .. (curPos + 1)] = -torch.nn.functional.cross_entropy (input: logits.transpose (1, 2),
                    target: tokens[.., (prevPos + 1) .. (curPos + 1)], reduction: torch.nn.Reduction.None, ignore_index: tokenizer.PadId);
            }

            if ((curPos - minPromptLen) % 5 == 0) {
                for (var i = 0; i < batch; i++) {
                    var toks = tokens[i][..(promptTokens[i].Length + maxGenLen)].data<long> ().Select (x => (int)x).ToArray ();
                    Console.WriteLine (i + ": " + string.Join (",", toks));
                    Console.WriteLine (i + ": " + tokenizer.Decode (toks));
                }
            }

            var newEosReached = eosReached | (~inputTextMask[.., curPos]) & (nextToken == tokenizer.EosId);

            // replace eosReached tensor
            eosReached.Dispose ();
            eosReached = newEosReached;
            scope.MoveToOuter (eosReached);

            if (eosReached.all ().item<bool> ()) {
                break;
            }

            prevPos = curPos;

            stopWatch.Stop ();
            Console.WriteLine ($"iteration {stopWatch.ElapsedMilliseconds} ms");
        }

        var outputTokens = new int[batch][];
        var outputLogProbs = new float[batch][];

        for (var i = 0; i < batch; i++) {
            // cut to max gen len
            var start = echo ? 0 : promptTokens[i].Length;
            var toks = tokens[i][start..(promptTokens[i].Length + maxGenLen)].data<long> ().Select (x => (int)x).ToArray ();
            float[]? probs = null;
            if (logProbs) {
                probs = tokenLogProbs![i][start..(promptTokens[i].Length + maxGenLen)].data<float> ().ToArray ();
            }

            // cut to first eos if any
            if (toks.Contains (tokenizer.EosId)) {
                var eosPos = Array.IndexOf (toks, tokenizer.EosId);
                toks = toks[..eosPos];
                if (logProbs) {
                    probs = probs![..eosPos];
                }
            }

            outputTokens[i] = toks;
            if (logProbs) {
                outputLogProbs[i] = probs!;
            }
        }

        return (outputTokens, logProbs ? null : outputLogProbs);
    }

    public static CompletionPrediction[] TextCompletion (
        Transformer transformer,
        ITokenizer tokenizer,
        string[] prompts,
        string device,
        int? maxGenLen = null,
        float temperature = 0.6f,
        float topP = 0.9f,
        bool logProbs = false,
        bool echo = false
    ) {
        maxGenLen ??= transformer.args.max_seq_len - 1;

        var prompTokens = prompts.Select (x => tokenizer.Encode (x, bos: true, eos: false)).ToArray ();
        var (outputTokens, outputLogProbs) = Generate (
            transformer,
            tokenizer,
            prompTokens,
            maxGenLen.Value,
            temperature,
            topP,
            logProbs,
            echo,
            device);
        return outputTokens
            .Select ((x, i) => new CompletionPrediction (tokenizer.Decode (x),
                x.Select (x => tokenizer.Decode ([x])).ToArray (),
                logProbs ? outputLogProbs![i] : null))
            .ToArray ();
    }

    static torch.Tensor SampleTopP (torch.Tensor logits, float topP, double temperature) {
        var probs = torch.softmax (logits[.., -1] / temperature, dim: -1);
        var (probsSort, probsIndex) = torch.sort (probs, dim: -1, descending: true);
        var cumsum = torch.cumsum (probsSort, dim: -1);
        var mask = cumsum - probsSort > topP;
        probsSort[mask] = 0f;
        probsSort /= probsSort.sum (dim: -1, keepdim: true);
        var nextToken = torch.multinomial (probsSort, num_samples: 1);
        nextToken = torch.gather (probsIndex, dim: -1, index: nextToken);
        return nextToken;
    }
}
