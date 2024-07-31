using System.Diagnostics;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace LLAMA;

public class Transformer : nn.Module<Tensor, int, Tensor>
{
    public ConfigurationParams args;

    Embedding tok_embeddings;

    ModuleList<nn.Module<Tensor, int, Tensor, Tensor?, Tensor>> layers;

    RMSNorm norm;

    Linear output;

    Tensor freqs_compex;

    public Transformer (ConfigurationParams args)
        : base (nameof(Transformer)) {
        Debug.Assert (args.vocab_size > 0, "vocab size must be set");

        this.args = args;
        this.tok_embeddings = nn.Embedding (
            args.vocab_size,
            this.args.dim,
            dtype: args.Dtype);

        this.layers = nn.ModuleList<nn.Module<Tensor, int, Tensor, Tensor?, Tensor>> ();
        for (int i = 0; i < args.n_layers; i++) {
            Console.WriteLine ("creating encoder block " + i);
            this.layers.Add (new EncoderBlock (args));
        }

        this.norm = new RMSNorm (args);
        this.output = nn.Linear (args.dim, args.vocab_size, dtype: args.Dtype, hasBias: false);
        this.freqs_compex = PrecomputeThetaPosFrequencies (args.dim / args.n_heads, args.max_seq_len * 2);

        RegisterComponents ();
    }

    public override Tensor forward (Tensor tokens, int startPos) {
        using var scope = NewDisposeScope ();

        // (B, Seq_Len) -> (B, Seq_Len, Dim)
        var seqLen = (int)tokens.shape[1];

        var h = this.tok_embeddings.forward (tokens);
        var freqsComplex = this.freqs_compex[startPos..(startPos + seqLen)].to (h.device);
        Tensor? mask = null;

        if (seqLen > 1) {
            var device = h.device;
            mask = full (new long[] {
                seqLen,
                seqLen
            }, dtype: ScalarType.Float32, value: float.NegativeInfinity, device: device);
            // (B, Seq_Len) -> (B, Seq_Len, Seq_Len)
            mask = triu (mask, diagonal: 1);

            if (device.type == DeviceType.MPS) {
                // BUG: https://github.com/pytorch/pytorch/issues/100005
                mask = nan_to_num (mask, 0.0);
            }

            // (B, Seq_Len, Seq_Len) -> (B, Seq_Len, Seq_Len)

            var zeros = torch.zeros (seqLen, startPos, device: device);
            mask = hstack ([zeros, mask]).type_as (h);
        }

        foreach (var layer in this.layers) {
            h = layer.forward (h, startPos, freqsComplex, mask);
        }

        // (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        h = this.norm.forward (h);

        // (B, Seq_Len, Dim) -> (B, Seq_Len, Vocab_Size)
        return scope.MoveToOuter (this.output.forward (h));
    }

    Tensor PrecomputeThetaPosFrequencies (int headDim, int seqLen, float theta = 10000.0f) {
        // As written in the paragraph 3.2.2 of the paper
        // >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
        Debug.Assert (headDim % 2 == 0, "Dimension must be divisible by 2");

        // Build the theta parameter
        // According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
        // Shape: (Head_Dim / 2)
        var thetaNumerator = arange (0, headDim, 2).to (float32);
        // Shape: (Head_Dim / 2)
        var thetaInput = pow (theta, -1.0f * (thetaNumerator / headDim)); // (Dim / 2)
        // Construct the positions (the "m" parameter)
        // Shape: (Seq_Len)
        var m = arange (seqLen);
        // Multiply each theta by each position using the outer product.
        // Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
        var freqs = outer (m, thetaInput)
            .to (float32, non_blocking: true);

        // We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
        // (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
        return polar (ones_like (freqs), freqs);
    }
}
