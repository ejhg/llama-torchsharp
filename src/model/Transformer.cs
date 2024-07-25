using System.Diagnostics;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace LLAMA;

public class Transformer : nn.Module<Tensor, int, Tensor>
{
    private ModelArgs args;
    private int vocabSize;
    private int nLayers;
    private Embedding tok_embeddings;
    private ModuleList<nn.Module<Tensor, int, Tensor, Tensor?, Tensor>> layers;
    private RMSNorm norm;
    private Linear output;
    private Tensor freqs_compex;

    public Transformer (ModelArgs args)
        : base (nameof(Transformer)) {
        Debug.Assert (args.VocabSize > 0, "vocab size must be set");

        this.args = args;
        this.vocabSize = args.VocabSize;
        this.nLayers = args.NLayers;
        this.tok_embeddings = nn.Embedding (this.vocabSize, this.args.Dim, dtype: args.Dtype);

        this.layers = nn.ModuleList<nn.Module<Tensor, int, Tensor, Tensor?, Tensor>> ();
        for (int i = 0; i < this.nLayers; i++) {
            Console.WriteLine ("creating encoder block " + i);
            this.layers.Add (new EncoderBlock (args));
        }

        this.norm = new RMSNorm (args);
        this.output = nn.Linear (args.Dim, this.vocabSize, dtype: args.Dtype, hasBias: false);
        RegisterComponents ();
        this.freqs_compex = PrecomputeThetaPosFrequencies (args.Dim / args.NHeads, args.MaxSeqLen * 2);
    }

    public ModelArgs Args => this.args;

    public override Tensor forward (Tensor tokens, int startPos) {
        // (B, Seq_Len) -> (B, Seq_Len, Dim)
        var batch = tokens.shape[0];
        var seqLen = (int)tokens.shape[1];

        // print tokens shape
        var h = this.tok_embeddings.forward (tokens);
        var freqsComplex = this.freqs_compex[startPos..(startPos + seqLen)].to (h.device);
        Tensor? mask = null;
        Console.WriteLine ($"tokens shape: {string.Join (",", tokens.shape)}");

        if (seqLen > 1) {
            var device = h.device;
            mask = torch.full (new long[] {
                seqLen,
                seqLen
            }, dtype: ScalarType.Float32, value: float.NegativeInfinity, device: device);
            // (B, Seq_Len) -> (B, Seq_Len, Seq_Len)
            mask = torch.triu (mask, diagonal: 1);
            // (B, Seq_Len, Seq_Len) -> (B, Seq_Len, Seq_Len)

            var zeros = torch.zeros (seqLen, startPos, device: device);
            mask = torch.hstack ([zeros, mask]).type_as (h);
        }

        for (int i = 0; i < this.nLayers; i++) {
            h = this.layers[i].forward (h, startPos, freqsComplex, mask);
        }


        // (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        h = this.norm.forward (h);
        // (B, Seq_Len, Dim) -> (B, Seq_Len, Vocab_Size)
        var output = this.output.forward (h);

        return output;
    }

    static Tensor PrecomputeThetaPosFrequencies (int headDim, int seqLen, float theta = 10000.0f) {
        // As written in the paragraph 3.2.2 of the paper
        // >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
        if (headDim % 2 != 0) {
            throw new ArgumentException ("Dimension must be divisible by 2", nameof(headDim));
        }

        // Build the theta parameter
        // According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
        // Shape: (Head_Dim / 2)
        var thetaNumerator = torch.arange (0, headDim, 2).to (torch.float32);
        // Shape: (Head_Dim / 2)
        var thetaInput = torch.pow (theta, -1.0f * (thetaNumerator / headDim)); // (Dim / 2)
        // Construct the positions (the "m" parameter)
        // Shape: (Seq_Len)
        var m = torch.arange (seqLen);
        // Multiply each theta by each position using the outer product.
        // Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
        var freqs = torch.outer (m, thetaInput).to (torch.float32);

        // We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
        // (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
        var freqsComplex = torch.polar (torch.ones_like (freqs), freqs);

        return freqsComplex;
    }
}
