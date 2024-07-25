using TorchSharp;
using TorchSharp.Modules;

namespace LLAMA;

public class FeedForward : torch.nn.Module<torch.Tensor, torch.Tensor>
{
    private Linear w1;
    private Linear w2;
    private Linear w3;

    public FeedForward (ModelArgs args)
        : base (nameof(FeedForward)) {
        var hiddenDim = args.Dim * 4;
        hiddenDim = 2 * hiddenDim / 3;
        hiddenDim = args.FFNDimMultiplier.HasValue ? (int)args.FFNDimMultiplier.Value * hiddenDim : hiddenDim;

        // Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hiddenDim = args.MultipleOf * ((hiddenDim + args.MultipleOf - 1) / args.MultipleOf);
        this.w1 = torch.nn.Linear (args.Dim, hiddenDim, hasBias: false, dtype: args.Dtype);
        this.w2 = torch.nn.Linear (hiddenDim, args.Dim, hasBias: false, dtype: args.Dtype);
        this.w3 = torch.nn.Linear (args.Dim, hiddenDim, hasBias: false, dtype: args.Dtype);

        RegisterComponents ();
    }

    public override torch.Tensor forward (torch.Tensor input) {
        // (B, Seq_Len, Dim) -> (B, Seq_Len, Hidden_Dim)
        var swish = torch.nn.functional.silu (this.w1.forward (input));
        // (B, Seq_Len, Hidden_Dim) -> (B, Seq_Len, Dim)
        var xV = this.w3.forward (input);
        // (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) -> (B, Seq_Len, Hidden_Dim)
        var x = swish * xV;
        // (B, Seq_Len, Hidden_Dim) -> (B, Seq_Len, Dim)
        x = this.w2.forward (x);

        return x;
    }
}
