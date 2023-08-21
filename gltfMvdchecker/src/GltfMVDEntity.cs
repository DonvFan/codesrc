using CBIMS.Geometry.Graphic.IfcSemantic;
using CBIMS.InterIFC.IFC;
using CBIMS.MVD.InterIfc;
using CBIMS.STEParser;
using System;
using System.Collections.Generic;
using System.Linq;
using CBIMS.MVD.InterIfc.STEP;
using CBIMS.GBGLTF.Gltf;

namespace CBIMS.MVD.Gltf
{
    public class GltfMVDEntity : IMvdEntity
    {
        protected readonly IEntity _entity;
        //protected readonly GltfEntity _entity;
        protected readonly EXPSchema _schema;
        public string IfcTypeName => _entity.IFC_TypeName;
        public GltfMVDEntity(IEntity entity, EXPSchema schema) 
        {
            _entity = entity;
            _schema = schema;
        }
        public GltfMVDEntity(IEntity entity)
        {
            _entity = entity;
            _schema = (entity.IFC_Model as GltfModel).Schema;
        }
        public MvdType MvdType => MvdType.ENTITY;

        

        protected virtual object GetOriginAttribute(string attrName)
        {
            try {

                if(attrName == "Tag" && _entity.ContainsAttr("Name"))
                {
                    string name = _entity["Name"] as string;
                    string[] tokens = name.Split(':');
                    int temp = 0;
                    if (tokens.Length >= 0 && int.TryParse(tokens[tokens.Length - 1], out temp)) 
                    {
                        return tokens[tokens.Length - 1];
                    }
                    return null;
                }
                else if (attrName == "GlobalId")
                {
                    return (_entity as GltfEntity).GlobalId;
                }
                if (_entity.ContainsAttr(attrName))
                {
                    return _entity[attrName];
                }
                else if(ContainsInv(attrName))
                {

                }
                else
                    return null;           
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.GetType().Name + ex.Message + ex.StackTrace);
            }
            return null;
            
        }

        internal bool ContainsInv(string attrName)
        {
            var arg = _schema.GetEXPEntity(_entity.IFC_TypeName).AllInverseArgs.FirstOrDefault(node => node.Name == attrName);
            if (arg == null)
                return false;
            return true;
        }



        public virtual IMvdType GetAttribute(string attrName)
        {
            var atrVal = GetOriginAttribute(attrName);
            return _getAttribute(attrName, atrVal, null);
        }

        private IMvdType _getAttribute(string attrName, object atrVal, string attrTypeName = null)
        {

            switch (atrVal)
            {
                case IEntity _:
                    return _build(atrVal, null);
                case ISelect asl:
                    return _getAttribute(attrName, asl.CLRData, asl.IFCDataType);
                case null:
                    return null;
                default:
                    if (attrTypeName == null)
                    {
                        var expEntity = _schema.GetEXPEntity(IfcTypeName);
                        var arg = expEntity.AllArgs.Concat(expEntity.AllInverseArgs).Where(a => a.Name == attrName).FirstOrDefault();
                        if (arg is EXPArgSingle argsingle)
                            attrTypeName = argsingle.Type.Name;
                        else if (arg is EXPArgInverse argInverse)
                            attrTypeName = argInverse.RelType;
                        else
                            throw new Exception();
                    }

                    return _build(atrVal, attrTypeName);
            }
        }

        private IMvdType _build(object entity, string ifcTypeName)
        {
            if (entity == null)
                return null;

            if (entity is IEnumerable<object> es && !(entity is string) && !(entity is IEntity))
            {
                var entities = new InterIfcMVDEntityCollection();
                foreach (var e in es)
                {
                    entities.Add(_build(e, ifcTypeName));
                }
                return entities;
            }
            else
            {

                switch (entity)
                {
                    case string sv:
                        return new InterIfcMVDString(sv, ifcTypeName);
                    case int iv:
                        return new InterIfcMVDInt(iv, ifcTypeName);
                    case bool bv:
                        return new InterIfcMVDBool(bv, ifcTypeName);
                    case double rv:
                        return new InterIfcMVDReal(rv, ifcTypeName);
                    case IEntity ev:
                        return new GltfMVDEntity(ev);
                    case IEnum ev:
                        return new InterIfcMVDString(ev.Name, ifcTypeName);
                    case ISelect sev:
                        return _build(sev.CLRData, sev.IFCDataType);
                    case InterIFC.STEP.STEPBoolean boo:
                        return new InterIfcMVDLogical(boo.Value, ifcTypeName);
                    default:
                        if (entity is bool?)
                        {
                            return new InterIfcMVDLogical((bool?)entity, ifcTypeName);
                        }
                        throw new Exception();
                }
            }
        }


        private string GetIfcSchemaVersion()
        {
            //return _entity.IFC_Model.STEPModel.Schema.Name;
            return _schema.Name;
        }

        public bool TryGetIfcSchemaVersion(out string schemaVersion)
        {
            schemaVersion = GetIfcSchemaVersion();
            return schemaVersion != null;
        }

        public int GetId() => _entity.IFC_Id;
        //{
        //    string name = _entity["Name"] as string;
        //    string[] tokens = name.Split(':');
        //    int res = _entity.IFC_Id;
        //    if(tokens.Length >= 0)
        //    {
        //        try 
        //        {
        //            res = int.Parse(tokens[tokens.Length - 1]);
        //        }
        //        catch
        //        {
        //            Console.WriteLine("string can't parse to int!");
        //        }
                
        //    }
        //    return res;
        //}

        public override int GetHashCode()
        {
            return _entity.GetHashCode();
        }

        public override bool Equals(object obj)
        {
            if(obj is GltfMVDEntity _ent)
                return _entity.Equals(_ent._entity);
            return false;
        }
    }
}
