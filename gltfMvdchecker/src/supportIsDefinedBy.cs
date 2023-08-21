using System;
using System.Collections.Generic;
using System.Text;
using CBIMS.Geometry.Graphic.IfcSemantic;
using CBIMS.InterIFC.IFC;
using CBIMS.InterIFC.STEP;

namespace CBIMS.MVD.Gltf
{
    public class Gltf_base : IEntity
    {
        private IModel _model { get; }
        private string _type_name { get; }
        public Dictionary<string, object> _attributes { get; set; }
        public Gltf_base(IModel model, string tname)
        {
            _model = model;
            _type_name = tname;
            IFC_Id = -1;
            _attributes = new Dictionary<string, object>();
        }

        public void addAttr(string k, object v)
        {
            if (!ContainsAttr(k))
                _attributes.Add(k, v);
        }


        public void changeAttr(string k, object v)
        {
            if (ContainsAttr(k))
                _attributes[k] = v;
        }

        public IModel IFC_Model => _model;
        public string IFC_TypeName => _type_name;
        public int IFC_Id { get; set; }
        public ISTEPEntity STEPEntity => throw new NotImplementedException();

        public object this[string attrName] => ContainsAttr(attrName) ? _attributes[attrName] : null;

        public bool ContainsAttr(string attrName)
        {
            return _attributes.ContainsKey(attrName);
        }
    }

    public class Gltf_IfcRelDefines : Gltf_base
    {
        public Gltf_IfcRelDefines(IModel model, string tname = "IfcRelDefines") :
            base(model, tname)
        {
            _attributes.Add("RelatedObjects", null);
        }
    }
    public class Gltf_IfcRelDefinesByProperties : Gltf_IfcRelDefines
    {
        public Gltf_IfcRelDefinesByProperties(IModel model, string tname = "IfcRelDefinesByProperties") :
            base(model, tname)
        {
            _attributes.Add("RelatingPropertyDefinition", null);
        }
    }

    public class Gltf_IfcRelDefinesByType : Gltf_IfcRelDefines
    {
        public Gltf_IfcRelDefinesByType(IModel model, string tname = "IfcRelDefinesByType") :
            base(model, tname)
        {

        }
    }


    public class Gltf_IfcPropertySetDefinition : Gltf_base
    {
        public Gltf_IfcPropertySetDefinition(IModel model, string tname = "IfcPropertySetDefinition") :
           base(model, tname)
        {

        }
    }

    public class Gltf_IfcPropertySet : Gltf_IfcPropertySetDefinition
    {
        public Gltf_IfcPropertySet(IModel model, string tname = "IfcPropertySet") :
            base(model, tname)
        {
            _attributes.Add("HasProperties", null);
        }
    }

    public class Gltf_IfcProperty : Gltf_base
    {
        public Gltf_IfcProperty(IModel model, string tname = "IfcProperty"):
            base(model, tname)
        {
            _attributes.Add("Name", null);
        }
    }

    public class Gltf_IfcSimpleProperty: Gltf_IfcProperty
    {
        public Gltf_IfcSimpleProperty(IModel model, string tname):
            base(model, tname)
        {
            _attributes.Add("NominalValue", null);
            _attributes.Add("Unit", null);
        }

    }

}
