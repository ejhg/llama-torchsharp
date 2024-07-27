using Microsoft.ML.Tokenizers;

namespace LLAMA;

public class TokenizeDecoder : TokenizerDecoder
{
    const char spaceReplacement = '▁';

    string bos = "<s>";

    string eos = "</s>";

    public TokenizeDecoder (string bos = "<s>", string eos = "</s>") {
        this.bos = bos;
        this.eos = eos;
    }

    public override string Decode (IEnumerable<string> tokens) {
        var str = string.Join ("", tokens);
        str = str.Replace (spaceReplacement, ' ');

        if (str.StartsWith (bos)) {
            str = str.Substring (bos.Length);
        }

        if (str.EndsWith (eos)) {
            str = str.Substring (0, str.Length - eos.Length);
        }

        return str;
    }
}
