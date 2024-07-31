using TorchSharp;
using TorchSharp.Modules;

namespace LLAMA;

public class SelfAttention : torch.nn.Module<torch.Tensor, int, torch.Tensor, torch.Tensor?, torch.Tensor>
{
    int nKVHeads;

    int nHeadsQ;

    /// <summary>
    /// Indicates how many times the Keys and Values should be repeated
    /// </summary>
    int nRep => this.nHeadsQ / this.nKVHeads;

    int headDim;

    Linear wq;

    Linear wk;

    Linear wv;

    Linear wo;

    torch.Tensor cache_k;

    torch.Tensor cache_v;

    public SelfAttention (ConfigurationParams args)
        : base (nameof(SelfAttention)) {
        // # Indicates the number of heads for the Keys and Values
        this.nKVHeads = args.n_kv_heads ?? args.n_heads;
        // Indicates the number of heads for the Queries
        this.nHeadsQ = args.n_heads;
        //Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        this.headDim = args.dim / args.n_heads;

        this.wq = torch.nn.Linear (args.dim, args.n_heads * this.headDim, hasBias: false, dtype: args.Dtype);
        this.wk = torch.nn.Linear (args.dim, this.nKVHeads * this.headDim, hasBias: false, dtype: args.Dtype);
        this.wv = torch.nn.Linear (args.dim, this.nKVHeads * this.headDim, hasBias: false, dtype: args.Dtype);
        this.wo = torch.nn.Linear (args.n_heads * this.headDim, args.dim, hasBias: false, dtype: args.Dtype);

        this.cache_k = torch.zeros (args.max_batch_size, args.max_seq_len, this.nKVHeads, this.headDim, dtype: args.Dtype);
        this.cache_v = torch.zeros (args.max_batch_size, args.max_seq_len, this.nKVHeads, this.headDim, dtype: args.Dtype);

        RegisterComponents ();
    }

    public override torch.Tensor forward (torch.Tensor input, int startPos, torch.Tensor freqsComplex, torch.Tensor? mask = null) {
        using var scope = torch.NewDisposeScope ();

        int batchSize = (int)input.shape[0];
        int seqLen = (int)input.shape[1];

        // (B, Seq_Len, Dim) -> (B, Seq_Len, N_Heads * Head_Dim)
        var xq = this.wq.forward (input);

        // (B, Seq_Len, Dim) -> (B, Seq_Len, H_KV * Head_Dim)
        var xk = this.wk.forward (input);

        // (B, Seq_Len, Dim) -> (B, Seq_Len, H_KV * Head_Dim)
        var xv = this.wv.forward (input);

        // (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xq = xq.view (batchSize, seqLen, this.nHeadsQ, this.headDim);

        // (B, Seq_Len, H_KV * Head_Dim) -> (B, Seq_Len, H_KV, Head_Dim)
        xk = xk.view (batchSize, seqLen, this.nKVHeads, this.headDim);

        // (B, Seq_Len, H_KV * Head_Dim) -> (B, Seq_Len, H_KV, Head_Dim)
        xv = xv.view (batchSize, seqLen, this.nKVHeads, this.headDim);
        // (B, Seq_Len, H_Q, Head_Dim) -> (B, Seq_Len, H_Q, Head_Dim)
        xq = ApplyRotaryEmbeddings (xq, freqsComplex);

        // (B, Seq_Len, H_KV, Head_Dim) -> (B, Seq_Len, H_KV, Head_Dim)
        xk = ApplyRotaryEmbeddings (xk, freqsComplex);
        // replace entries in cache
        this.cache_k[..batchSize, startPos..(startPos + seqLen)] = xk;
        this.cache_v[..batchSize, startPos..(startPos + seqLen)] = xv;

        var keys = this.cache_k[..batchSize, ..(startPos + seqLen)];
        var values = this.cache_v[..batchSize, ..(startPos + seqLen)];

        // Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.
        // (B, Seq_Len, H_KV, Head_Dim) -> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = RepeatKV (keys, this.nRep);

        // (B, Seq_Len, H_KV, Head_Dim) -> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = RepeatKV (values, this.nRep);

        // (B, Seq_Len, H_Q, Head_Dim) -> (B, H_Q, Seq_Len, Head_Dim)
        xq = xq.transpose (1, 2);

        // (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose (1, 2);

        // (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose (1, 2);
        // (B, H_Q, Seq_Len, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, Seq_Len, Seq_Len_KV)
        var scores = torch.matmul (xq, keys.transpose (2, 3)) / Math.Sqrt (this.headDim);
        if (mask is not null) {
            scores = scores + mask;
        }

        var softmax = torch.nn.functional.softmax (scores, dim: -1);

        // (B, H_Q, Seq_Len, Seq_Len_KV) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, Seq_Len, Head_Dim)
        var output = torch.matmul (softmax, values);

        // (B, H_Q, Seq_Len, Head_Dim) -> (B, Seq_Len, H_Q, Head_Dim) -> (B, Seq_Len, Dim)
        output = output.transpose (1, 2).contiguous ().view (batchSize, seqLen, -1);

        // (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        output = this.wo.forward (output);

        return scope.MoveToOuter (output);
    }

    static torch.Tensor RepeatKV (torch.Tensor x, int nRep) {
        var batchSize = x.shape[0];
        var seqLen = x.shape[1];
        var nKVHeads = x.shape[2];
        var headDim = x.shape[3];
        if (nRep == 1) {
            return x;
        }

        return x.unsqueeze (3)
            .expand (batchSize, seqLen, nKVHeads, nRep, headDim)
            .reshape (batchSize, seqLen, nKVHeads * nRep, headDim);
    }

    static torch.Tensor ApplyRotaryEmbeddings (torch.Tensor input, torch.Tensor freqsComplex) {
        // Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
        // Two consecutive values will become a single complex number
        // (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
        var input_complex = input
            .to_type (torch.ScalarType.Float32, non_blocking: true)
            .reshape (input.shape[0], input.shape[1], input.shape[2], -1, 2)
            .view_as_complex ();

        // Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
        // (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
        var freqs_complex_reshaped = freqsComplex
            .unsqueeze (0)
            .unsqueeze (2);

        // Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
        // Which results in the rotation of the complex number as shown in the Figure 1 of the paper
        // (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
        var rotated_complex = input_complex * freqs_complex_reshaped;
        // Console.WriteLine(rotated_complex.mean().ToSingle());

        // Convert the complex number back to the real number
        // (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
        var rotated = rotated_complex.view_as_real ();

        // (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
        var rotated_reshaped = rotated.reshape (rotated.shape[0], rotated.shape[1], rotated.shape[2], -1);

        return rotated_reshaped.to_type (input.dtype, non_blocking: true);
    }
}
