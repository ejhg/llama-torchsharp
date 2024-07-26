using TorchSharp;

namespace llama.unpickler;

static class DelayedExcecutionLoader
{
    public static void optimized_load_py (
        this torch.nn.Module module,
        string location
    ) {
        using FileStream fileStream = File.OpenRead (location);

        using (torch.NewDisposeScope ()) {
            using (torch.no_grad ()) {
                var hashtable = DelayedExecutionUnpickler.UnpickleStateDict (fileStream, leaveOpen: true);
                Dictionary<string, Func<object[]>> source = new();

                foreach (string key in hashtable.Keys) {
                    source.Add (key, (Func<object[]>)hashtable[key]);
                }

                load_state_dict (module, source);

                fileStream.Close ();
            }
        }
    }

    static void load_state_dict (
        torch.nn.Module module,
        Dictionary<string, Func<object[]>> source
    ) {
        var dictionary = module.state_dict ();

        using (torch.no_grad ()) {
            foreach (string key in source.Keys) {
                if (!dictionary.ContainsKey (key)) {
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

                using torch.Tensor temp = torch.WrappedTensorDisposeScope (() => torch
                    .empty (shape, tObject.dtype)
                    .as_strided (shape, stride, storageOffset));

                using var stream = tObject.data;
                temp.ReadBytesFromStream (stream);
                stream.Close ();

                Console.WriteLine ($"loading {key} [{string.Join (",", shape)}] {tObject.dtype} -> {dictionary[key].dtype}");

                dictionary[key].copy_ (temp);
            }
        }
    }
}
