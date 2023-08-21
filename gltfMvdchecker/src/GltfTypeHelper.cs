using CBIMS.InterIFC.IFC;
using CBIMS.STEParser;
using CBIMS.Geometry.Graphic.IfcSemantic;
using CBIMS.GBGLTF.Gltf;
using System;
using System.Collections.Generic;
using System.Linq;


namespace CBIMS.MVD.Gltf
{
    public class GltfTypeHelper : IMvdTypeHelper
    {
        private readonly GltfModel _model; //GltfMVDModel???
        private readonly EXPSchema _expHelper;
        private readonly IList<GltfEntity> _entities;

        public GltfTypeHelper(string ifcSchemaVersion, IModel model) : 
            this((IfcSchemaVersion)Enum.Parse(typeof(IfcSchemaVersion), ifcSchemaVersion), model) 
        {
        }
        public GltfTypeHelper(IfcSchemaVersion ifcSchemaVersion, IModel model)
        { 
            _expHelper = EXPSchema.GetEXPSchema(ifcSchemaVersion); 
            _model = model as GltfModel;
            //Console.WriteLine(typeof(IIfcProduct).Name);
            _entities = _model.FindAllEntities(typeof(IIfcProduct));
            //relRe = new Dictionary<string, Dictionary<GltfEntity, HashSet<GltfEntity>>>();
            //relnames = new HashSet<string>();
            //getAllInvs();
        }


        public IReadOnlyCollection<IMvdEntity> GetEntitiesByType(string typeName)
        {
            //var _entities = _model.FindAllEntities<IIfcProduct>();
            List<IMvdEntity> mvdEntities = new List<IMvdEntity>();
            if (typeName.StartsWith("Ifc"))
            {
                typeName = typeName.ToUpper();
            }

            if (typeName == "IFCPRODUCT")
            {
                foreach (var e in _entities)
                {
                    mvdEntities.Add(new GltfMVDEntity(e, _expHelper));
                }
            }
            else
            {
                HashSet<string> typeNames = new HashSet<string>(_expHelper.GetSubTypeNames(typeName));
                typeNames.Add(typeName);
                
                foreach(var e in _entities)
                {
                    if (typeNames.Contains(e.IFC_TypeName))
                        mvdEntities.Add(new GltfMVDEntity(e, _expHelper));
                }
            }

            return mvdEntities;
        }

        bool ContainsInv(GltfEntity _entity, string attrName)
        {
            var arg = _expHelper.GetEXPEntity(_entity.IFC_TypeName).AllInverseArgs.FirstOrDefault(node => node.Name == attrName);
            if (arg == null)
                return false;
            return true;
        }
        public IMvdEntity  GetEntityById(int id)
        {
            return new GltfMVDEntity(_model.GetEntity(id) as GltfEntity, _expHelper);
        }

        public bool IsSubTypeOf(string typeName, string targetTypeName)
        {
            return _expHelper.IsSubTypeOf(typeName, targetTypeName);
        }

        public bool IsTypeOf(string typeName, string targetTypeName)
        {
            return _expHelper.IsTypeOf(typeName, targetTypeName);
        }

        public ICollection<IMvdEntity> GetEntitiesByIds(ICollection<int> ids)
        {
            List<IMvdEntity> output = new List<IMvdEntity>();
            
            foreach(var id in ids)
            {
                var res = _model.GetEntity(id) as GltfEntity;
                if (res != null)
                    output.Add(new GltfMVDEntity(res, _expHelper));
            }

            return output;
        }

        public ICollection<Tuple<IMvdEntity, IMvdType>> BatchGetAttributes(ICollection<IMvdEntity> entities, string attr)
        {
            //Console.WriteLine(attr);

            List<Tuple<IMvdEntity, IMvdType>> output = new List<Tuple<IMvdEntity, IMvdType>>();
                foreach (var entity in entities)
                {
                    var mvdType = entity.GetAttribute(attr);
                    if (mvdType != null)
                    {
                        output.Add(new Tuple<IMvdEntity, IMvdType>(entity, mvdType));
                    }
                }

                return output;

        }
    }
}
