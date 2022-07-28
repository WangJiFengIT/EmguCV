using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Demo1
{
    class Program
    {
        static void Main(string[] args)
        {
            test();
            //TemplateMatching();
        }

        static void test()
        {
            Mat src = CvInvoke.Imread("E:\\workspace\\EmguCV\\Demo1\\images\\b.jpg", LoadImageType.AnyColor);//从本地读取图片
            Mat result = src.Clone();

            Mat tempImg = CvInvoke.Imread("E:\\workspace\\EmguCV\\Demo1\\images\\temp1.jpg", LoadImageType.AnyColor);
            int matchImg_rows = src.Rows - tempImg.Rows + 1;
            int matchImg_cols = src.Cols - tempImg.Cols + 1;
            Mat matchImg = new Mat(matchImg_rows, matchImg_cols, DepthType.Cv32F, 1); //存储匹配结果
            #region 模板匹配参数说明
            ////采用系数匹配法，匹配值越大越接近准确图像。
            ////IInputArray image：输入待搜索的图像。图像类型为8位或32位浮点类型。设图像的大小为[W, H]。
            ////IInputArray templ：输入模板图像，类型与待搜索图像类型一致，并且大小不能大于待搜索图像。设图像大小为[w, h]。
            ////IOutputArray result：输出匹配的结果，单通道，32位浮点类型且大小为[W - w + 1, H - h + 1]。
            ////TemplateMatchingType method：枚举类型标识符，表示匹配算法类型。
            ////Sqdiff = 0 平方差匹配，最好的匹配为 0。
            ////SqdiffNormed = 1 归一化平方差匹配，最好效果为 0。
            ////Ccorr = 2 相关匹配法，数值越大效果越好。
            ////CcorrNormed = 3 归一化相关匹配法，数值越大效果越好。
            ////Ccoeff = 4 系数匹配法，数值越大效果越好。
            ////CcoeffNormed = 5 归一化系数匹配法，数值越大效果越好。
            #endregion
            CvInvoke.MatchTemplate(src, tempImg, matchImg, TemplateMatchingType.CcoeffNormed);
            #region 归一化函数参数说明
            ////IInputArray src：输入数据。
            ////IOutputArray dst：进行归一化后输出数据。
            ////double alpha = 1; 归一化后的最大值，默认为 1。
            ////double beta = 0：归一化后的最小值，默认为 0。
            #endregion
            //CvInvoke.Normalize(matchImg, matchImg, 0, 1, NormType.MinMax, matchImg.Depth); //归一化
            CvInvoke.Normalize(matchImg, matchImg, 255, 0, Emgu.CV.CvEnum.NormType.MinMax);
            double minValue = 0.0, maxValue = 0.0;
            Point minLoc = new Point();
            Point maxLoc = new Point();
            #region 极值函数参数说明
            ////IInputArray arr：输入数组。
            ////ref double minVal：输出数组中的最小值。
            ////ref double maxVal; 输出数组中的最大值。
            ////ref Point minLoc：输出最小值的坐标。
            ////ref Point maxLoc; 输出最大值的坐标。
            ////IInputArray mask = null：蒙版。
            #endregion
            CvInvoke.MinMaxLoc(matchImg, ref minValue, ref maxValue, ref minLoc, ref maxLoc);

            //绘制矩形，匹配得到的效果。
            CvInvoke.Rectangle(src, new Rectangle(maxLoc, tempImg.Size), new MCvScalar(0, 0, 255), 3);

            CvInvoke.Imshow("result", src);
            CvInvoke.WaitKey(0);
        }



        /// <summary>
        /// 模板匹配
        /// </summary>
        static void TemplateMatching()
        {
            Mat src = CvInvoke.Imread("E:\\workspace\\EmguCV\\Demo1\\images\\template.png", LoadImageType.AnyColor);
            CvInvoke.Imshow("src", src);
            Mat result = src.Clone();

            Mat tempImg = CvInvoke.Imread("E:\\workspace\\EmguCV\\Demo1\\images\\matching.jpg", LoadImageType.AnyColor);
            int matchImg_rows = src.Rows - tempImg.Rows + 1;
            int matchImg_cols = src.Cols - tempImg.Cols + 1;
            //存储匹配结果
            Mat matchImg = new Mat(matchImg_rows, matchImg_cols, DepthType.Cv32F, 1);
            //采用系数匹配法，匹配值越大越接近准确图像。
            //IInputArray image：输入待搜索的图像。图像类型为8位或32位浮点类型。设图像的大小为[W, H]。
            //IInputArray templ：输入模板图像，类型与待搜索图像类型一致，并且大小不能大于待搜索图像。设图像大小为[w, h]。
            //IOutputArray result：输出匹配的结果，单通道，32位浮点类型且大小为[W - w + 1, H - h + 1]。
            //TemplateMatchingType method：枚举类型标识符，表示匹配算法类型。
            //Sqdiff = 0 平方差匹配，最好的匹配为 0。
            //SqdiffNormed = 1 归一化平方差匹配，最好效果为 0。
            //Ccorr = 2 相关匹配法，数值越大效果越好。
            //CcorrNormed = 3 归一化相关匹配法，数值越大效果越好。
            //Ccoeff = 4 系数匹配法，数值越大效果越好。
            //CcoeffNormed = 5 归一化系数匹配法，数值越大效果越好。
            CvInvoke.MatchTemplate(src, tempImg, matchImg, TemplateMatchingType.CcoeffNormed);
            CvInvoke.Imshow("match", matchImg);
            //归一化函数
            CvInvoke.Normalize(matchImg, matchImg, 0, 1, NormType.MinMax, matchImg.Depth);
            double minValue = 0.0, maxValue = 0.0;
            Point minLoc = new Point();
            Point maxLoc = new Point();
            CvInvoke.MinMaxLoc(matchImg, ref minValue, ref maxValue, ref minLoc, ref maxLoc);
            Image<Gray, Single> imgMatch = matchImg.ToImage<Gray, Single>();
            int count = 0;
            int tempH = 0, tempW = 0;
            for (int i = 0; i < imgMatch.Rows; i++)
            {
                for (int j = 0; j < imgMatch.Cols; j++)
                {
                    float matchValue = imgMatch.Data[i, j, 0];
                    if ((matchValue > 0.7) && (Math.Abs(j - tempW) > 10) && (Math.Abs(i - tempH) > 10))  //只绘制处于最大范围内的点
                    {
                        count++;
                        CvInvoke.Rectangle(result, new Rectangle(j, i, tempImg.Width, tempImg.Height), new MCvScalar(255, 0, 0), 2);
                        tempH = i;
                        tempW = j;
                    }
                }
            }
            CvInvoke.Imshow("result", result);
            CvInvoke.WaitKey(0);
        }

        /// <summary>
        /// 行人检测
        /// </summary>
        static void PedDetection()
        {
            string imgPath = "E:\\workspace\\EmguCV\\Demo1\\images\\5.jpg";
            Image<Bgr, Byte> image = new Image<Bgr, byte>(imgPath);
            MCvObjectDetection[] regions;
            using (HOGDescriptor des = new HOGDescriptor())
            {
                des.SetSVMDetector(HOGDescriptor.GetDefaultPeopleDetector());
                //获取图片行人特征
                regions = des.DetectMultiScale(image);
            }
            //绘制图片特征
            foreach (MCvObjectDetection pedestrain in regions)
            {
                image.Draw(pedestrain.Rect, new Bgr(Color.Red), 2);
            }
            CvInvoke.Imshow("image", image);
            CvInvoke.WaitKey(0);
        }

        static void showText()
        {
            string win1 = "Hello World!";
            //新建窗口
            CvInvoke.NamedWindow(win1);
            //新建图像
            Mat img = new Mat(200, 500, DepthType.Cv8U, 3);
            //设置图像颜色
            img.SetTo(new Bgr(255, 0, 0).MCvScalar);
            //绘制文字
            CvInvoke.PutText(img, "wang ji jifeng test", new System.Drawing.Point(10, 80), FontFace.HersheyComplex, 2.0, new Bgr(0, 255, 255).MCvScalar);
            //显示
            CvInvoke.Imshow(win1, img);
            CvInvoke.WaitKey(0);
            CvInvoke.DestroyWindow(win1);
        }
    }
}
