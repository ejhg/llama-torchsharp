using TorchSharp;
using TorchSharp.Modules;

namespace LLAMA;

public class RMSNorm : torch.nn.Module<torch.Tensor, torch.Tensor>
{
    float _eps;

    Parameter weight;

    public RMSNorm (ConfigurationParams args)
        : base (nameof(RMSNorm)) {
        this._eps = args.norm_eps;

        // the gamma scalar
        this.weight = torch.nn.Parameter (torch.ones (args.dim, dtype: args.Dtype));
    }

    public override torch.Tensor forward (torch.Tensor input) {
        using var scope = torch.NewDisposeScope ();

        var x = input
            // needs higher precision for the norm so convert to float32
            .to_type (torch.ScalarType.Float32, non_blocking: true);
        // (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        var normed = x * torch.rsqrt (x.pow (2).mean ([-1L], keepdim: true) + this._eps);

        // (B, Seq_Len, Dim) * (Dim) = (B, Seq_Len, Dim)
        return scope.MoveToOuter (this.weight * normed.to_type (input.dtype, non_blocking: true));
    }
}
