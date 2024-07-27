using Microsoft.ML.Tokenizers;

namespace LLAMA;

public class PreTokenizer : Microsoft.ML.Tokenizers.PreTokenizer
{
    public override IReadOnlyList<Split> PreTokenize (string sentence) {
        var split = new Split (sentence, new(0, sentence.Length));

        return new List<Split> { split };
    }
}
