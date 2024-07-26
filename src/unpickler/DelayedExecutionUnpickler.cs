using Razorvine.Pickle;
using Razorvine.Pickle.Objects;
using System.Collections;
using System.IO.Compression;
using TorchSharp;

namespace llama.unpickler
{
    public static class DelayedExecutionUnpickler
    {
        public class TensorStream
        {
            public Stream data { get; set; }

            public torch.ScalarType dtype { get; set; }
        }

        static DelayedExecutionUnpickler () {
            Unpickler.registerConstructor ("torch._utils", "_rebuild_tensor", new TensorObjectConstructor ());
            Unpickler.registerConstructor ("torch._utils", "_rebuild_tensor_v2", new TensorObjectConstructor ());
            Unpickler.registerConstructor ("torch._utils", "_rebuild_parameter", new ParameterObjectConstructor ());
            Unpickler.registerConstructor ("collections", "OrderedDict", new OrderedDictObjectConstructor ());
        }

        public static Hashtable UnpickleStateDict (Stream stream, bool leaveOpen = false) {
            byte[] buffer = new byte[4];
            stream.Read (buffer, 0, 4);

            if (buffer[0] != (byte)80 || buffer[1] != (byte)75 || buffer[2] != (byte)3 || buffer[3] != (byte)4) {
                throw new NotSupportedException ("Unsupported file format");
            }

            stream.Seek (0L, SeekOrigin.Begin);

            using ZipArchive archive = new(stream, ZipArchiveMode.Read, leaveOpen);
            ZipArchiveEntry zipArchiveEntry = archive.Entries.First (e => e.Name.EndsWith ("data.pkl"));

            return (Hashtable)new CustomUnpickler (archive).load (zipArchiveEntry.Open ());
        }

        class CustomUnpickler : Unpickler
        {
            readonly ZipArchive _archive;

            public CustomUnpickler (ZipArchive archive) => this._archive = archive;

            protected override object persistentLoad (object pid) {
                var objArray = (object[])pid;
                var storage = (string)objArray[0] == "storage"
                    ? ((ClassDictConstructor)objArray[1]).name
                    : throw new NotImplementedException ("Unknown persistent id loaded");
                var archiveKey = (string)objArray[2];
                var typeFromStorageName = GetScalarTypeFromStorageName (storage);
                var zipArchiveEntry = this._archive.Entries
                    .First (f => f.FullName.EndsWith ("data/" + archiveKey));

                return new TensorStream {
                    data = zipArchiveEntry.Open (),
                    dtype = typeFromStorageName
                };
            }

            static torch.ScalarType GetScalarTypeFromStorageName (string storage) {
                return storage switch {
                    "IntStorage" => torch.@int,
                    "CharStorage" => torch.int8,
                    "HalfStorage" => torch.half,
                    "LongStorage" => torch.@long,
                    "BoolStorage" => torch.@bool,
                    "ByteStorage" => torch.uint8,
                    "FloatStorage" => torch.@float,
                    "ShortStorage" => torch.int16,
                    "DoubleStorage" => torch.float64,
                    "BFloat16Storage" => torch.bfloat16,
                    "ComplexFloatStorage" => torch.cfloat,
                    "ComplexDoubleStorage" => torch.cdouble,
                    _ => throw new NotImplementedException ()
                };
            }
        }

        class OrderedDict : Hashtable
        {
            public void __setstate__ (Hashtable arg) {
                foreach (string key in (IEnumerable)arg.Keys) {
                    if ((object)(arg[(object)key] as torch.Tensor) != null)
                        this[(object)key] = arg[(object)key];
                }
            }
        }

        class OrderedDictObjectConstructor : IObjectConstructor
        {
            public object construct (object[] args) => (object)new OrderedDict ();
        }

        class TensorObjectConstructor : IObjectConstructor
        {
            public object construct (object[] args) {
                var delayed = () => args;

                return delayed;
            }
        }

        class ParameterObjectConstructor : IObjectConstructor
        {
            public object construct (object[] args) {
                throw new NotImplementedException ();
            }
        }
    }
}
