import settings;
import three;
import solids;unitsize(4cm);

currentprojection=perspective( camera = (1.2, 1.0, 0.5), target = (0.0, 0.0, 0.0) );
currentlight=nolight;

revolution S=sphere(O,1);
pen SpherePen = rgb(0.85,0.85,0.85)+opacity(0.6);
pen SphereLinePen = rgb(0.75,0.75,0.75)+opacity(0.6)+linewidth(0.5pt);
draw(surface(S), surfacepen=SpherePen, meshpen=SphereLinePen);

/*
  Colors
*/
pen curveStyle1 = rgb(0.0,0.6,0.5333333333333333)+linewidth(0.75pt)+opacity(1.0);
pen pointStyle1 = rgb(0.0,0.0,0.0)+linewidth(3.5pt)+opacity(1.0);
pen pointStyle2 = rgb(0.0,0.4666666666666667,0.7333333333333333)+linewidth(3.5pt)+opacity(1.0);

/*
  Exported Points
*/
dot( (1.0,0.0,0.0), pointStyle1);
dot( (0.11061587104123714,0.11061587104123713,0.9876883405951378), pointStyle1);
dot( (0.0,1.0,0.0), pointStyle1);
dot( (0.7071067811865476,0.7071067811865475,0.0), pointStyle2);
dot( (-0.7071067811865476,-0.7071067811865475,-0.0), pointStyle2);

/*
  Exported Curves
*/
path3 p1 = (0.11061587104123714,0.11061587104123713,0.9876883405951378) .. (0.20791790791406004,0.20791790791406,0.9557930147983302) .. (0.3010714243544671,0.301071424354467,0.9048270524660196) .. (0.3882177576326267,0.38821775763262656,0.8358073613682704) .. (0.46761810444947915,0.46761810444947904,0.7501110696304596) .. (0.5376882147304866,0.5376882147304864,0.6494480483301837) .. (0.5970300016676451,0.5970300016676449,0.5358267949789968) .. (0.6444594373045622,0.6444594373045621,0.4115143586051089) .. (0.6790301770727896,0.6790301770727895,0.2789911060392293) .. (0.7000524419064874,0.7000524419064874,0.14090123193758283) .. (0.7071067811865476,0.7071067811865475,1.1102230246251565e-16);
 draw(p1, curveStyle1);