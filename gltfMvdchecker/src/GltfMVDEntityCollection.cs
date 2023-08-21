using System;
using System.Collections.Generic;
using System.Text;

namespace CBIMS.MVD.Gltf
{
    sealed public class GltfMVDEntityCollection : IMvdEntityCollection
    {
        private readonly List<IMvdType> _entityCollection;

        public GltfMVDEntityCollection(IEnumerable<IMvdType> es) => _entityCollection = new List<IMvdType>(es);

        public GltfMVDEntityCollection(int capacity) => _entityCollection = new List<IMvdType>(capacity);

        public GltfMVDEntityCollection() => _entityCollection = new List<IMvdType>();

        public IReadOnlyCollection<IMvdType> AsEnumerable() => _entityCollection;

        public string IfcTypeName => throw new NotSupportedException();

        public MvdType MvdType => MvdType.LIST;

        public bool TryGetIfcSchemaVersion(out string schemaVersion)
        {
            schemaVersion = null;
            return false;
        }

        public void Add(IMvdType entity) => _entityCollection.Add(entity);
    }
}
