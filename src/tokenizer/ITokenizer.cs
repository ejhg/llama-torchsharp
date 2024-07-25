namespace LLAMA;

public interface ITokenizer
{
    public int[] Encode (string input, bool bos, bool eos);

    public string Decode (int[] input);

    public int VocabSize { get; }

    public int PadId { get; }

    public int EosId { get; }
}
