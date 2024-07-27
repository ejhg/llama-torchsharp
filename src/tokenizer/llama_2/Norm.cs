using Microsoft.ML.Tokenizers;

namespace LLAMA;

public class Norm : Normalizer
{
    public override NormalizedString Normalize (string original) {
        // replace space with _
        var normalized = original.Replace (" ", "▁");

        return new NormalizedString (original, normalized, null, isOneToOneMapping: true);
    }
}
