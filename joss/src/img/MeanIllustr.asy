import settings;
import three;
import solids;unitsize(4cm);

currentprojection=perspective( camera = (0.7, 0.7, 0.5), target = (0.0, 0.0, 0.0) );
currentlight=nolight;

revolution S=sphere(O,0.995);
pen SpherePen = rgb(0.85,0.85,0.85)+opacity(0.6);
pen SphereLinePen = rgb(0.75,0.75,0.75)+opacity(0.6)+linewidth(0.5pt);
draw(surface(S), surfacepen=SpherePen, meshpen=SphereLinePen);

/*
  Colors
*/
pen pointStyle1 = rgb(0.9333333333333333,0.4666666666666667,0.2)+linewidth(3.5pt)+opacity(1.0);
pen pointStyle2 = rgb(0.2,0.7333333333333333,0.9333333333333333)+linewidth(2.5pt)+opacity(1.0);
pen pointStyle3 = rgb(0.0,0.6,0.5333333333333333)+linewidth(3.5pt)+opacity(1.0);

/*
  Exported Points
*/
dot( (0.6276622839456928,0.5525902203153832,0.5483467021181695), pointStyle1);
dot( (0.6183928035520221,0.5693777980721136,0.5416634227798861), pointStyle2);
dot( (0.7788182407488278,0.42136761235386855,0.4646412413207647), pointStyle3);
dot( (0.6349296362650473,0.6065157129726909,0.4785426281007402), pointStyle3);
dot( (0.26189249205033827,0.9634175986407473,-0.05691092370503037), pointStyle3);
dot( (0.6369229087385905,0.409421989430216,0.6532249558115316), pointStyle3);
dot( (0.5566536105085663,-0.4706972238498739,0.6845296789532205), pointStyle3);
dot( (0.24550212373244118,0.834349698291047,0.4935476554543648), pointStyle3);
dot( (0.04291558551193708,0.5042963085628513,0.8624636141252854), pointStyle3);
dot( (0.6516831317796825,0.7198366549506655,0.2390487103986586), pointStyle3);
dot( (0.6388740074864943,0.7670357391724101,-0.05912848205714044), pointStyle3);
dot( (0.6363843126193481,0.731346355354713,0.24524990348902842), pointStyle3);
dot( (0.6381579302094254,0.16350101922792337,0.7523442515379861), pointStyle3);
dot( (0.75949325809956,0.5449888596082848,0.3551860551938615), pointStyle3);
dot( (0.4071987703483637,-0.8853890576030367,0.2242217163960427), pointStyle3);
dot( (0.8684189152900917,0.0966381772060067,0.48632257841136806), pointStyle3);
dot( (0.4896452488000417,0.6858131851592795,0.5384308733618728), pointStyle3);
dot( (0.4122848961786072,0.7268149064786432,0.5493280041136046), pointStyle3);
dot( (0.2647968409861693,0.6600512593932353,0.703004244636652), pointStyle3);
dot( (0.8378044071234323,0.12047710355233943,0.532512011999911), pointStyle3);
dot( (0.1354576083407774,0.7579535729616944,0.6380890357757276), pointStyle3);
dot( (0.44653738206389837,0.4608881694076247,0.7669331533579762), pointStyle3);
dot( (0.12982305061719107,0.9734042269448693,0.18875959974080211), pointStyle3);
dot( (0.36329986043885176,0.775450596714363,0.5164199681077887), pointStyle3);
dot( (0.49258599835851197,0.18180065263676853,0.8510626045844063), pointStyle3);
dot( (0.012460894543859624,0.8811371253661362,0.4726966177250149), pointStyle3);
dot( (0.836433846325503,0.535886470355055,0.11490914503000066), pointStyle3);
dot( (0.1755425904858794,0.5001784267039845,0.8479424157249322), pointStyle3);
dot( (0.7478038281590013,0.45177332564153233,0.4865082700525585), pointStyle3);
dot( (0.4608935557979982,0.2788964703678542,0.842492664086893), pointStyle3);
dot( (0.5630902645754734,0.6632323190257472,0.4930032910032683), pointStyle3);
dot( (0.5653231188748902,0.7921114479540892,0.2301504405506175), pointStyle3);
dot( (0.8870167515951841,0.010974261727141998,0.4616068110081047), pointStyle3);
dot( (0.4998881270012441,0.5571723412104812,0.6630767999811329), pointStyle3);
dot( (0.7246619908600336,0.6362271175789806,0.2647263754519278), pointStyle3);
dot( (0.8529206386118746,-0.13433391803687678,0.5044608832157099), pointStyle3);
dot( (0.9081445377096014,0.41218914461915185,0.07330489537777685), pointStyle3);
dot( (0.6790872237251572,0.5679572037258195,0.4650431778977161), pointStyle3);
dot( (0.2924248952552447,0.8882076161122173,0.35436550525581784), pointStyle3);
dot( (0.5417779503186889,0.5686679728144094,0.618945384701777), pointStyle3);
dot( (0.934941627504318,0.2780781104060356,0.22035588867236156), pointStyle3);
dot( (0.2120892412792208,0.904641524453446,0.3696510056366), pointStyle3);