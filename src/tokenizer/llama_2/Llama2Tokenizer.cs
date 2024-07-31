using Microsoft.ML.Tokenizers;

namespace LLAMA;

public class Llama2Tokenizer : ITokenizer
{
    private Tokenizer tokenizer;

    private bool addPrecedingSpace;

    public int VocabSize => this.tokenizer.Model.GetVocabSize ();

    public int PadId { get; }

    int BosId { get; }

    public int EosId { get; }

    public Llama2Tokenizer (
        string vocabPath,
        string mergesPath,
        bool addPrecedingSpace = true,
        int padToken = -1,
        int startToken = 1,
        int endToken = 2
    ) {
        this.BosId = startToken;
        this.EosId = endToken;
        this.addPrecedingSpace = addPrecedingSpace;
        this.PadId = padToken;

        var model = new Bpe (vocabPath, mergesPath);
        this.tokenizer = new Tokenizer (
            model,
            preTokenizer: new PreTokenizer (),
            normalizer: new Norm ()) {
            Decoder = new TokenizeDecoder (
                model.IdToToken (this.BosId)!,
                model.IdToToken (this.EosId)!)
        };
    }

    public string Decode (int[] input) {
        var str = this.tokenizer.Decode (input) ?? throw new Exception ("Failed to decode");
        if (this.addPrecedingSpace) {
            str = str.TrimStart ();
        }

        return str;
    }

    public int[] Encode (string input, bool bos, bool eos) {
        if (this.addPrecedingSpace) {
            input = " " + input;
        }

        var tokens = this.tokenizer.Encode (input).Ids.ToArray ();
        if (bos) {
            tokens = new[] { this.BosId }.Concat (tokens).ToArray ();
        }

        if (eos) {
            tokens = tokens.Concat (new[] { this.EosId }).ToArray ();
        }

        Console.WriteLine ($"tokens: {string.Join (",", tokens)}");

        return tokens;
    }
}
