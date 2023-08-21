using System;
using System.IO;
using CBIMS.Geometry.Graphic.IfcSemantic;
using CBIMS.MVDLite.Checker;

namespace CBIMS.MVD.Gltf
{
    public class test
    {
        static void Main(string[] args)
        {
            string gltfPath = "F:\\projects\\CIBMS\\testdata\\小测试用例模型.glb";
            string mvdPath = "F:\\projects\\CIBMS\\testdata\\小规模模型标准检查规则.mvdlite";
            string pathOut = "F:\\projects\\CIBMS\\testdata\\out.xlsx";
            MvdLiteReportGltf(gltfPath, mvdPath, pathOut);
        }
        
        static void MvdLiteReportGltf(string gltfFilePath, string mvdPath, string reportFilePath, bool multiThread = false)
        {

            GltfModel model = new GltfModel(gltfFilePath);
            IMvdTypeHelper typeHelper = new GltfTypeHelper(model.SchemaVersion, model);
            MVDLiteChecker checker = new MVDLiteChecker(typeHelper);


            checker.logger = new MVDConsoleLogger();

            checker.ReadMVDLite(mvdPath);


            var checkStartTime = DateTime.Now;

            checker.ModelName = Path.GetFileName(gltfFilePath);
            checker.MvdName = Path.GetFileName(mvdPath);
            checker.CheckTime = checkStartTime.ToString();

            checker.Check();


            var checkEndTime = DateTime.Now;
            var checkConsume = checkEndTime - checkStartTime;

            checker.CheckTimeConsume = checkConsume.TotalSeconds.ToString() + " s";


            if (reportFilePath.EndsWith(".xlsx"))
            {
                checker.SaveResult_XLSX(reportFilePath);
            }
            else if (reportFilePath.EndsWith(".md"))
            {
                checker.SaveResult_MD(reportFilePath);
            }
            else
            {
                checker.SaveResult_MD(reportFilePath);
            }

        }

    }
}