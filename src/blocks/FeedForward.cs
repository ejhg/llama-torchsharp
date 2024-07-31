using TorchSharp;
using TorchSharp.Modules;

namespace LLAMA;

public class FeedForward : torch.nn.Module<torch.Tensor, torch.Tensor>
{
    Linear w1;

    Linear w2;

    Linear w3;

    public FeedForward (ConfigurationParams args)
        : base (nameof(FeedForward)) {
        var hiddenDim = args.dim * 4;
        hiddenDim = 2 * hiddenDim / 3;
        hiddenDim = args.ffn_dim_multiplier.HasValue
            ? (int)(args.ffn_dim_multiplier.Value * hiddenDim)
            : hiddenDim;

        // Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hiddenDim = args.multiple_of * ((hiddenDim + args.multiple_of - 1) / args.multiple_of);
        this.w1 = torch.nn.Linear (args.dim, hiddenDim, hasBias: false, dtype: args.Dtype);
        this.w2 = torch.nn.Linear (hiddenDim, args.dim, hasBias: false, dtype: args.Dtype);
        this.w3 = torch.nn.Linear (args.dim, hiddenDim, hasBias: false, dtype: args.Dtype);

        RegisterComponents ();
    }

    public override torch.Tensor forward (torch.Tensor input) {
        using var scope = torch.NewDisposeScope ();

        // (B, Seq_Len, Dim) -> (B, Seq_Len, Hidden_Dim)
        var swish = torch.nn.functional.silu (this.w1.forward (input));
        // (B, Seq_Len, Hidden_Dim) -> (B, Seq_Len, Dim)
        var xV = this.w3.forward (input);
        // (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) -> (B, Seq_Len, Hidden_Dim)
        var x = swish * xV;
        // (B, Seq_Len, Hidden_Dim) -> (B, Seq_Len, Dim)
        x = this.w2.forward (x);

        return scope.MoveToOuter (x);
    }
}
