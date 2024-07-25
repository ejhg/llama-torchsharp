using TorchSharp;

namespace LLAMA;

public class EncoderBlock : torch.nn.Module<torch.Tensor, int, torch.Tensor, torch.Tensor?, torch.Tensor>
{
    private SelfAttention attention;
    private FeedForward feed_forward;
    private RMSNorm attention_norm;
    private RMSNorm ffn_norm;

    public EncoderBlock (JsonModelArgs args)
        : base (nameof(EncoderBlock)) {
        this.attention = new SelfAttention (args);
        this.feed_forward = new FeedForward (args);
        this.attention_norm = new RMSNorm (args);
        this.ffn_norm = new RMSNorm (args);
    }

    public override torch.Tensor forward (torch.Tensor input, int startPos, torch.Tensor freqsComplex, torch.Tensor? mask) {
        // (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        var x = this.attention_norm.forward (input);
        // (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        x = this.attention.forward (x, startPos, freqsComplex, mask);
        // (B, Seq_Len, Dim) + (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        var h = x + input;
        // (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        x = this.ffn_norm.forward (h);
        // (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        x = this.feed_forward.forward (x);
        // (B, Seq_Len, Dim) + (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        x = x + h;

        return x;
    }
}
