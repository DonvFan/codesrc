using CBIMS.Core;
using CBIMS.Core.Base;
using CBIMS.Core.Geometry;
using CBIMS.InterIFC.IFC;
using CBIMS.InterIFC.STEP;
using System.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using CBIMS.Geometry.Graphic.IfcSemantic;
using CBIMS.Geometry.Interop;
using System.Numerics;
using CBIMS.Geometry.Graphic.BuildingStorey;
using Newtonsoft.Json;
using CBIMS.Geometry.Graphic.Gltf;

namespace CBIMS.MVD.Gltf
{
    //public class t1 : GltfModel
    //{
    //    public t1(string path): base(path)
    //    {
    //        base.Entities
    //    }
    //}

    public class GltfMVDModel
    {

        private Dictionary<int, GltfEntity> _entities;
        private Dictionary<int, DataIfcMaterial> _materials;
        private Dictionary<int, CbimShapeGeometry> _shapeGeoms;
        //private Dictionary<int, Matrix4x4> _placements;
        private Dictionary<int, CbimRect3D> _bboxes;
        private List<GltfIfcProperty> _properties;
        private List<GltfIfcProperty> _quantities;
        private List<GltfIfcRelationship> _relationships;
        private List<string> _labels;
        private double _modelBase;
        private CbimRect3D _bbox;
        // private StoreyInformation _storey;

        // if this GltfMVDModel is indeed associated with an ifcModel
        private IModel _model;
        // calculate hash key from ifc-json string, if is an ifcModel, calculate from ifcModel instead
        private string _ifcHash;
        private string _gltfHash;
        // the geometry store of this GltfMVDModel, write all mesh data into binary
        private IGeometryStore _store;
        // use default modelFactor for now, but it should be written into the ifc-json
        private ModelFactor _factor = new ModelFactor();
        // progress indicator
        private IProgress<int> _progress;

        public ICollection<IEntity> GetEntities(ICollection<int> Ids)
        {
            throw new NotImplementedException();
        }
        public GltfMVDModel(string gltfPath, IProgress<int> progress = null)
        {
            _progress = progress;
            _progress?.Report(1);
            // open gltf file with handler
            IfcGltfHandler _gltfHandler = new IfcGltfHandler();
            _gltfHandler.Import(gltfPath);
            // if the gltf file is valid
            if (_gltfHandler.IfcJson?.GLTF?.SceneIfcProject != null)
            {
                _progress?.Report(10);
                // update hash key
                _gltfHash = CalculateGltfHashKey(_gltfHandler.IfcJsonString);

                var gltf = _gltfHandler.IfcJson.GLTF;
                // @todo: transform every node into GltfEntity
                _entities = new Dictionary<int, GltfEntity>();
                GltfEntity projectNode = new GltfEntity(this, gltf.SceneIfcProject);
                foreach (var node in gltf.SceneIfcProject.Nodes)
                {
                    GltfEntity nodeEntity = new GltfEntity(this, node);
                    projectNode.Nodes.Add(nodeEntity);
                }
                _progress?.Report(50);
                // geometry store
                InitGeometryStore(_gltfHandler);
                _progress?.Report(70);
                // storey information
                //_storey = new StoreyInformation();
                // _storey.InitByGltf(this);
                // properties
                _properties = gltf.Properties;
                _quantities = gltf.Quantities == null ? new List<GltfIfcProperty>() : gltf.Quantities;
                _labels = gltf.Labels;
                // relationships
                _relationships = gltf.Relationships.Where(r => r.Relating > 0 && r.Related?.Count > 0).ToList();
                _progress?.Report(90);
            }
        }

        private void InitGeometryStore(IfcGltfHandler _gltfHandler)
        {
            // init things
            var gltf = _gltfHandler.IfcJson.GLTF;
            _shapeGeoms = new Dictionary<int, CbimShapeGeometry>();
            _materials = new Dictionary<int, DataIfcMaterial>();
            _bboxes = new Dictionary<int, CbimRect3D>();
            // transfer everything into a geometry store
            _store = new InMemoryGeometryStore();
            // getting the model base while making out the outter bound
            float tempModelBase = float.PositiveInfinity;
            float minPosX = float.PositiveInfinity;
            float minPosY = float.PositiveInfinity;
            float minPosZ = float.PositiveInfinity;
            float maxPosX = float.NegativeInfinity;
            float maxPosY = float.NegativeInfinity;
            float maxPosZ = float.NegativeInfinity;
            Matrix4x4 rotationHere = Matrix4x4.CreateRotationX((float)Math.PI / 2);
            using (var storeInit = _store.BeginInit())
            {
                // handle mesh data
                for (int i = 0; i < gltf.MeshList.Count; i++)
                {
                    CbimShapeGeometry shapeGeom = gltf.MeshList[i].ToCbimsShapeGeometry();
                    int shapeLabel = storeInit.AddShapeGeometry(shapeGeom);
                    shapeGeom.ShapeLabel = shapeLabel;
                    _shapeGeoms.Add(i, shapeGeom);
                }
                // handle style data
                for (int i = 0; i < gltf.MaterialList.Count; i++)
                {
                    _materials.Add(i, gltf.MaterialList[i]);
                }
                // @todo: handle mesh with vertex color

                // handle representation DataIfcNode
                foreach (var entity in _entities.Values)
                {
                    // calculate its bounding box
                    double pMinPosX = double.PositiveInfinity;
                    double pMinPosY = double.PositiveInfinity;
                    double pMinPosZ = double.PositiveInfinity;
                    double pMaxPosX = double.NegativeInfinity;
                    double pMaxPosY = double.NegativeInfinity;
                    double pMaxPosZ = double.NegativeInfinity;
                    if (entity.HasRepresentation)
                    {
                        foreach (var rep in entity.Representations)
                        {
                            var theGeom = _shapeGeoms[rep.MeshPrimitive.MeshIndex];
                            var theTran = new Matrix4x4(
                                (float)rep.Transformation[0],
                                (float)rep.Transformation[1],
                                (float)rep.Transformation[2],
                                (float)rep.Transformation[3],
                                (float)rep.Transformation[4],
                                (float)rep.Transformation[5],
                                (float)rep.Transformation[6],
                                (float)rep.Transformation[7],
                                (float)rep.Transformation[8],
                                (float)rep.Transformation[9],
                                (float)rep.Transformation[10],
                                (float)rep.Transformation[11],
                                (float)rep.Transformation[12],
                                (float)rep.Transformation[13],
                                (float)rep.Transformation[14],
                                (float)rep.Transformation[15]
                                );
                            var theFinalTran = theTran * rotationHere;
                            CbimsShapeInstance instance = new CbimsShapeInstance
                            {
                                IfcProductLabel = entity.IFC_Id,
                                ShapeGeometryLabel = theGeom.ShapeLabel,
                                StyleLabel = rep.MeshPrimitive.MaterialIndex + 1,
                                RepresentationType = CbimsGeometryRepresentationType.OpeningsAndAdditionsIncluded,
                                RepresentationContext = -1, // no context info in gltf yet
                                IfcTypeId = entity.IFC_TypeName,
                                Transformation = theFinalTran,
                                BoundingBox = theGeom.BoundingBox
                            };
                            storeInit.AddShapeInstance(instance, theGeom.ShapeLabel);
                            // update bounding box and model base
                            var bb0 = theGeom.BoundingBox;
                            var bb = bb0.Transform(theFinalTran);
                            tempModelBase = tempModelBase < bb.Min.Z ? tempModelBase : bb.Min.Z;
                            minPosX = minPosX < bb.Min.X ? minPosX : bb.Min.X;
                            minPosY = minPosY < bb.Min.Y ? minPosY : bb.Min.Y;
                            minPosZ = minPosZ < bb.Min.Z ? minPosZ : bb.Min.Z;
                            maxPosX = maxPosX > bb.Max.X ? maxPosX : bb.Max.X;
                            maxPosY = maxPosY > bb.Max.Y ? maxPosY : bb.Max.Y;
                            maxPosZ = maxPosZ > bb.Max.Z ? maxPosZ : bb.Max.Z;
                            pMinPosX = pMinPosX < bb.Min.X ? pMinPosX : bb.Min.X;
                            pMinPosY = pMinPosY < bb.Min.Y ? pMinPosY : bb.Min.Y;
                            pMinPosZ = pMinPosZ < bb.Min.Z ? pMinPosZ : bb.Min.Z;
                            pMaxPosX = pMaxPosX > bb.Max.X ? pMaxPosX : bb.Max.X;
                            pMaxPosY = pMaxPosY > bb.Max.Y ? pMaxPosY : bb.Max.Y;
                            pMaxPosZ = pMaxPosZ > bb.Max.Z ? pMaxPosZ : bb.Max.Z;
                        }
                        var p_bbox = new CbimRect3D(new Vector3(minPosX, minPosY, minPosZ), new Vector3(maxPosX, maxPosY, maxPosZ));
                        _bboxes.Add(entity.IFC_Id, p_bbox);
                    }
                }
                storeInit.Commit();
            }
            if (!double.IsInfinity(tempModelBase))
                _modelBase = tempModelBase;
            _bbox = new CbimRect3D(new Vector3(minPosX, minPosY, minPosZ), new Vector3(maxPosX, maxPosY, maxPosZ));
        }

        public string AssociateIfcModel(IModel ifcModel)
        {
            _model = ifcModel;
            _ifcHash = CalculateIfcHashKey();
            return IfcUID;
        }
        //public StoreyInformation Storey => _storey;
        public double ModelBase => _modelBase;
        public CbimRect3D BoundingBox => _bbox;
        public Dictionary<int, CbimRect3D> BoundingBoxMap => _bboxes;
        public string UID => _gltfHash;
        internal string CalculateGltfHashKey(string jsonString)
        {
            string hashkey = "";
            using (SHA1 sha = SHA1.Create())
            {
                using (MemoryStream tempStream = new MemoryStream(/*length?*/))
                {
                    using StreamWriter sw = new StreamWriter(tempStream);
                    sw.Write(jsonString);
                    tempStream.Position = 0;
                    var hash = sha.ComputeHash(tempStream);
                    hashkey = System.Convert.ToBase64String(hash);
                }
            }
            return hashkey;
        }
        public string IfcUID => _ifcHash;
        internal string CalculateIfcHashKey()
        {
            string hashkey = "";
            using (SHA1 sha = SHA1.Create())
            {
                using (MemoryStream tempStream = new MemoryStream(/*length?*/))
                {
                    _model.Write(tempStream);
                    tempStream.Position = 0;
                    var hash = sha.ComputeHash(tempStream);
                    hashkey = System.Convert.ToBase64String(hash);
                }
            }
            return hashkey;
        }

        public IGeometryStore Store => _store;


        public ModelFactor Factor => _factor;
        public string ToBCJson()
        {
            var jsonObj = new GltfLoader().LoadModel(this);

            return JsonConvert.SerializeObject(jsonObj); ;
        }

        public ISTEPModel STEPModel => _model?.STEPModel;

        public CBIMS.InterIFC.IFC.IfcSchemaVersion SchemaVersion =>
            _model == null ? CBIMS.InterIFC.IFC.IfcSchemaVersion.IFC2X3 : _model.SchemaVersion;

        public IReadOnlyCollection<IEntity> Entities => _entities.Values;

        // @todo: how to find sub classes
        public IList<GltfEntity> FindAllEntities<T>(bool findSubClasses = true) where T : IEntity
        {
            List<T> result = new List<T>();
            Type TT = typeof(T);

            string typeName = TT.Name;
            if (TT.IsInterface && typeName.StartsWith("IIfc"))
            {
                typeName = typeName.Substring(1).ToUpper();
            }
            if (typeName == "IFCPRODUCT")
                return _entities?.Values?.Where(v => !this.DefaultExclusions().Contains(v.IFC_TypeName)).ToList();
            return _entities?.Values?.Where(e => e.IFC_TypeName == typeName)?.ToList();
        }

        // @todo: how to find sub classes
        public IList<GltfEntity> FindAllEntities(Type TT, bool findSubClasses = true)
        {
            string typeName = TT.Name;
            if (TT.IsInterface && typeName.StartsWith("IIfc"))
            {
                typeName = typeName.Substring(1).ToUpper();
            }
            if (typeName == "IFCPRODUCT")
                return _entities?.Values?.ToList();
            return _entities?.Values?.Where(e => e.IFC_TypeName == typeName)?.ToList();
        }

        // @ignore
        public IList<T> FindAll<T>(bool findSubClasses = true) where T : IEntity
        {
            throw new NotImplementedException();
        }

        // @ignore
        public IList<IEntity> FindAll(Type TT, bool findSubClasses = true)
        {
            throw new NotImplementedException();
        }

        // @ignore
        public IList<T> FindInverse<T>(IEntity ent, string relArg) where T : IEntity
        {
            throw new NotImplementedException();
        }

        public void AddEntity(GltfEntity entity, int id = -1)
        {
            if (_entities == null)
                _entities = new Dictionary<int, GltfEntity>();
            if (id != -1)
                _entities[id] = entity;
            else
                _entities[entity.IFC_Id] = entity;
        }

        public IEntity GetEntity(int Id)
        {
            if (_entities?.Count > 0)
            {
                if (_entities.ContainsKey(Id))
                {
                    return _entities[Id];
                }
            }
            return null;
        }

        public void RemoveEntity(int Id)
        {
            if (_entities?.Count > 0)
            {
                if (_entities.ContainsKey(Id))
                {
                    _entities.Remove(Id);
                }
            }
        }

        public DataIfcMaterial GetMaterial(int Id)
        {
            var index = Id - 1;
            if (_materials?.Count > 0)
            {
                if (_materials.ContainsKey(index))
                {
                    return _materials[index];
                }
            }
            return new DataIfcMaterial();
        }

        // will write the json string into a given stream
        public void Write(Stream stream)
        {
            throw new NotImplementedException();
        }

        public void Write(string path)
        {
            throw new NotImplementedException();
        }

        public Matrix4x4 GetPlacement(GltfEntity entity)
        {
            return Matrix4x4.Identity; // identity for now, as we haven't write the entire placement tree to gltf yet
        }

        public GltfIfcProperty GetProperty(int index)
        {
            return index < _properties.Count ? _properties[index] : null;
        }

        public GltfIfcProperty GetQuantity(int index)
        {
            return index < _quantities.Count ? _quantities[index] : null;
        }

        public string GetLabel(int index)
        {
            return index < _labels.Count ? _labels[index] : string.Empty;
        }

        public GltfIfcRelationship GetRelationship(int index)
        {
            return index < _relationships.Count ? _relationships[index] : null;
        }

        public IEntity STEPToEntity(ISTEPEntity stepent)
        {
            throw new NotImplementedException();
        }

        public ISelect STEPToSelect(string nameUpper, ISTEPArgument steparg)
        {
            throw new NotImplementedException();
        }
    }
}

