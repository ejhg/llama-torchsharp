using TorchSharp;
using TorchSharp.Modules;

namespace LLAMA;

public class RMSNorm : torch.nn.Module<torch.Tensor, torch.Tensor>
{
    private float _eps;
    private Parameter weight;

    public RMSNorm (ModelArgs args)
        : base (nameof(RMSNorm)) {
        this._eps = args.NormEps;

        // the gamma scalar
        this.weight = torch.nn.Parameter (torch.ones (args.Dim, dtype: args.Dtype));
    }

    private torch.Tensor Norm (torch.Tensor x) {
        // (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        // rsqrt = 1 / sqrt
        var output = x * torch.rsqrt (x.pow (2).mean ([-1L], keepdim: true) + this._eps);
        return output;
    }

    public override torch.Tensor forward (torch.Tensor input) {
        // needs higher precision for the norm so convert to float32
        // (B, Seq_Len, Dim)
        var normed = this.Norm (input.to_type (torch.ScalarType.Float32)).type_as (input);
        // (B, Seq_Len, Dim) * (Dim) = (B, Seq_Len, Dim)
        var output = this.weight * normed;

        return output;
    }
}
