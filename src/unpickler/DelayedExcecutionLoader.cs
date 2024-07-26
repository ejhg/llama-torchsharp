using TorchSharp;

namespace llama.unpickler;

static class DelayedExcecutionLoader
{
    public static void optimized_load_py (
        this torch.nn.Module module,
        string location,
        Dictionary<string, bool> loadedParameters
    ) {
        using FileStream fileStream = File.OpenRead (location);

        using (torch.NewDisposeScope ()) {
            using (torch.no_grad ()) {
                var hashtable = DelayedExecutionUnpickler.UnpickleStateDict (fileStream, leaveOpen: true);
                Dictionary<string, Func<object[]>> source = new();

                foreach (string key in hashtable.Keys) {
                    source.Add (key, (Func<object[]>)hashtable[key]);
                }

                var unexpectedKeyes = load_state_dict (module, source).unexpected_keyes;

                fileStream.Close ();

                foreach (string key in hashtable.Keys) {
                    loadedParameters[key] = true;
                }

                foreach (string key in unexpectedKeyes) {
                    loadedParameters[key] = false;
                }
            }
        }
    }

    static (IList<string> missing_keys, IList<string> unexpected_keyes) load_state_dict (
        torch.nn.Module module,
        Dictionary<string, Func<object[]>> source
    ) {
        List<string> missing = new();
        List<string> unexpected = new();

        Dictionary<string, torch.Tensor> dictionary = module.state_dict ();

        foreach (string key in source.Keys) {
            if (!dictionary.ContainsKey (key))
                unexpected.Add (key);
        }

        foreach (string key in dictionary.Keys) {
            if (!source.ContainsKey (key))
                missing.Add (key);
        }

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

                dictionary[key].copy_ (temp);
            }

            return (missing, unexpected);
        }
    }
}
