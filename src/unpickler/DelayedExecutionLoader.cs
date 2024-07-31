using TorchSharp;

namespace llama.unpickler;

static class DelayedExecutionLoader
{
    public static void load_weights (
        this torch.nn.Module module,
        string location
    ) {
        using FileStream fileStream = File.OpenRead (location);

        using var scope = torch.NewDisposeScope ();
        using var no_grad = torch.no_grad ();

        var hashtable = DelayedExecutionUnpickler.UnpickleStateDict (fileStream, leaveOpen: true);
        Dictionary<string, Func<object[]>> source = new();

        foreach (string key in hashtable.Keys) {
            source.Add (key, (Func<object[]>)hashtable[key]);
        }

        load_state_dict (module, source);

        fileStream.Close ();
    }

    static void load_state_dict (
        torch.nn.Module module,
        Dictionary<string, Func<object[]>> source
    ) {
        var state_dict = module.state_dict ();

        foreach (string key in source.Keys) {
            if (!state_dict.ContainsKey (key)) {
                continue;
            }

            var args = source[key] ();

            var storageOffset = (int)args[1];
            var shape = ((object[])args[2])
                .Select ((Func<object, long>)(i => (int)i))
                .ToArray ();
            var stride = ((object[])args[3])
                .Select ((Func<object, long>)(i => (int)i))
                .ToArray ();

            var tObject = ((DelayedExecutionUnpickler.TensorStream)args[0]);

            using var stream = tObject.data;

            if (tObject.dtype == state_dict[key].dtype) {
                // read directly into target.
                var target = state_dict[key];
                target
                    .as_strided (shape, stride, storageOffset)
                    .ReadBytesFromStream (stream);
                stream.Close ();
            } else {
                // type conversion required. load onto cpu first before copying to target.
                using torch.Tensor temp = torch
                    .empty (shape, tObject.dtype, device: torch.CPU)
                    .as_strided (shape, stride, storageOffset);
                temp.ReadBytesFromStream (stream);
                stream.Close ();
                state_dict[key].copy_ (temp);
            }

            Console.WriteLine ($"loading {key} [{string.Join (",", shape)}] {tObject.dtype} -> {state_dict[key].dtype}");
        }
    }
}
