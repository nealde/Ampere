
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
// #include "ida/ida.h"
// #include "ida/ida_dense.h"
// #include "ida/nvector_serial.h"
// #include "ida/sundials_math.h"
// #include "ida/sundials_types.h"
#include "../ida/ida.h"
#include "../ida/ida_dense.h"
#include "../ida/nvector_serial.h"
#include "../ida/sundials_math.h"
#include "../ida/sundials_types.h"

// Problem Constants
// #define NEQ   190
#define NOUT  3000

#define ZERO RCONST(0.0);
#define ONE  RCONST(1.0);

// Prototypes of functions called by IDA
int resrob(realtype tres, N_Vector yy, N_Vector yp,
           N_Vector resval, void *user_data);

// activate when using stop condition //
static int grob(realtype t, N_Vector yy, N_Vector yp,
                realtype *gout, void *user_data);

// Prototypes of private functions
static void PrintOutput(void *mem, realtype t, N_Vector y);
static int check_flag(void *flagvalue, char *funcname, int opt);

//--------------------------------------------------------------------
// Main Program
//--------------------------------------------------------------------
int cc = 1;
int N1 = 7;
int N2 = 3;
int N3 = 7;
int Nr1 = 3;
int Nr2 = 3;
int NEQ = 5;
int to_print = 1;
// model constants
double F = 96487.0;
double R = 8.3143;
// double TC = 30.0;

double p[10000];

int drive(double* input, double* output, int n)
{
  void     *mem;
  N_Vector  yy, yp, avtol, id;
  realtype  rtol, *uv, *upv, *atval, *idv, *yprint;
  realtype  dt, t0, tout1, tout, tret;
  realtype  MapleGenVar1, MapleGenVar2;
  int       iout, retval, retvalr;
  int       rootsfound[1];
  int       step, count;
  int       err = 0;
  long      int i;
  double*   x_init = NULL;
  double*   r      = NULL;
  double*   tf     = NULL;
  // FILE     *savefile, *savetime, *filetime;
  // FILE     *ic_out;
  // clock_t  startt, endt;

  // startt = clock();

  mem    = NULL;
  yy     = yp  = id  = avtol = NULL;
  uv     = upv = idv = atval = NULL;
  yprint = NULL;


  for (i=0; i<n; i++){
      p[i] = input[i];
      // printf("%i ", i);
      // printf("%15.4e \n", p[i]);
    }
    // cc = p[27];
    // int initial = p[28];


    N1 = (int) p[25];
    N2 = (int) p[26];
    N3 = (int) p[27];
    Nr1 = (int) p[28];
    Nr2 = (int) p[29];
    // NEQ = 192;
    NEQ = 5*N1+2*N2+5*N3+N1*Nr1+N3*Nr2+14;

  // Allocate N-vectors.
  yy = N_VNew_Serial(NEQ);
  if(check_flag((void *)yy   , "N_VNew_Serial", 0)) return(1);
  yp = N_VNew_Serial(NEQ);
  if(check_flag((void *)yp   , "N_VNew_Serial", 0)) return(1);
  avtol = N_VNew_Serial(NEQ);
  if(check_flag((void *)avtol, "N_VNew_Serial", 0)) return(1);
  id = N_VNew_Serial(NEQ);
  if(check_flag((void *)id   , "N_VNew_Serial", 0)) return(1);

  // Create and initialize  y, y', and absolute tolerance vectors.
     uv = NV_DATA_S(yy);
    upv = NV_DATA_S(yp);
    idv = NV_DATA_S(id);
  atval = NV_DATA_S(avtol);

  rtol = RCONST(1.000000e-04);

  for (i = 0; i < NEQ; i++)
  {
     uv[i] = 0.0;
    upv[i] = 0.0;
    idv[i] = 0.0;
  atval[i] = RCONST(1.000000e-05);
  }

  // input uv
  // Integration limits
    t0 = 0.000000;
    dt = 10.0;
    tout1 = 10.0;
    // tfinal is now
     double tfinal = dt*NOUT;
     if (p[33] > 0){
         tfinal = p[33]+200.0;
     }
       cc = p[31];
       int initial = (int) p[32];
           if (initial == 1){
  count = 0;
  for (int i=0; i<N1+N2+N3+4; i++){
    uv[count] = 0.99998+2e-6*i;
    count++;
  }
  // uv[0] = 0.999982600307166;
  // uv[8] = 0.999995206282564;
  // uv[15] = 0.100000503236079E1;
  // uv[26] = 0.100001694344342E1;

  // uv[1] = 0.999982587379252;
  // uv[2] = 0.999982547462381;
  // uv[3] = 0.999982480152111;
  // uv[4] = 0.99998238537944;
  // uv[5] = 0.999982266536442;
  // uv[6] = 0.999982262814386;
  // uv[7] = 0.999982623989203;
  // uv[9] = 0.999999402609166;
  // uv[10] = 0.999998668938011;
  // uv[11] = 0.99999981512526;
  // uv[12] = 0.100000020333399E1;
  // uv[13] = 0.100000143072035E1;
  // uv[14] = 0.100000044557599E1;
  // uv[16] = 0.100001685368058E1;
  // uv[17] = 0.100001717469159E1;
  // uv[18] = 0.100001737080152E1;
  // uv[19] = 0.10000172787347E1;
  // uv[20] = 0.100001718880833E1;
  // uv[21] = 0.100001711296441E1;
  // uv[22] = 0.100001705128047E1;
  // uv[23] = 0.100001700362941E1;
  // uv[24] = 0.10000169698961E1;
  // uv[25] = 0.100001694999643E1;
  for (i=0; i<N1+N2+N3+3; i++){
    uv[count] = -.3447E-2+.01E-2*i;
    count++;
  }
  count++;
  // uv[27] = -0.344748854839739E-2;
  // uv[28] = -0.342837360376479E-2;
  // uv[29] = -0.337102876986701E-2;
  // uv[30] = -0.327536649824175E-2;
  // uv[31] = -0.314123898207062E-2;
  // uv[32] = -0.296843797612545E-2;
  // uv[33] = -0.275668996258097E-2;
  // uv[34] = -0.250565668979802E-2;
  // uv[35] = -0.221455818360745E-2;
  // uv[36] = -0.210375762111702E-2;
  // uv[37] = -0.199311844964017E-2;
  // uv[38] = -0.188241772266993E-2;
  // uv[39] = -0.177174183082613E-2;
  // uv[40] = -0.166103844611685E-2;
  // uv[41] = -0.155040750573269E-2;
  // uv[42] = -0.143959416518745E-2;
  // uv[43] = -0.118774398810412E-2;
  // uv[44] = -0.960834678291093E-3;
  // uv[45] = -0.758306483922333E-3;
  // uv[46] = -0.579998520200354E-3;
  // uv[47] = -0.425755295450909E-3;
  // uv[48] = -0.29545095069945E-3;
  // uv[49] = -0.18897967142332E-3;
  // uv[50] = -0.106255679087519E-3;
  // uv[51] = -0.472124456596996E-4;
  // uv[52] = -0.118031506938721E-4;
  for (i=0; i<N1+2; i++){
    uv[count] = 0.422461225901562E1;
    count++;
  }
  for (i=0; i<N3+2; i++){
    uv[count] = 0.822991162960124E-1;
    count++;
  }
  // uv[54] = 0.422461225901562E1;
  // uv[55] = 0.422461372095497E1;
  // uv[56] = 0.422461499109508E1;
  // uv[57] = 0.422461606899646E1;
  // uv[58] = 0.422461695391777E1;
  // uv[59] = 0.422461764481462E1;
  // uv[60] = 0.422461814033787E1;
  // uv[61] = 0.422461843883136E1;
  // uv[62] = 0.42246185383292E1;
  // uv[63] = 0.8229904651456E-1;
  // uv[64] = 0.822991162960124E-1;
  // uv[65] = 0.822993256403693E-1;
  // uv[66] = 0.822996734836089E-1;
  // uv[67] = 0.823001588804393E-1;
  // uv[68] = 0.823007810022684E-1;
  // uv[69] = 0.823015391353989E-1;
  // uv[70] = 0.823024326795067E-1;
  // uv[71] = 0.823034611463678E-1;
  // uv[72] = 0.823046241588292E-1;
  // uv[73] = 0.823059214500159E-1;
  // uv[74] = 0.823073528627725E-1;
  // negative particle - center, then internal, then surface
  for (i=0; i<N3*(Nr2+1); i++){
    uv[count] = 0.986699999999968;
    count++;
  }
  for (i=0; i<N3; i++){
    uv[count] = 0.977101677061948;
    count++;
  }
  // uv[75] = 0.986700000093982;
  // uv[76] = 0.986700000093982;
  // uv[77] = 0.986700000093982;
  // uv[78] = 0.986700000093982;
  // uv[79] = 0.986700000093982;
  // uv[80] = 0.986700000093982;
  // uv[81] = 0.986700000093982;


  // uv[82] = 0.986700000093982;
  // uv[83] = 0.986700000093982;
  // uv[84] = 0.986700000093982;
  // uv[85] = 0.9867;
  // uv[86] = 0.9867;
  // uv[87] = 0.9867;
  // uv[88] = 0.9867;
  // uv[89] = 0.9867;
  // uv[90] = 0.9867;
  // uv[91] = 0.9867;
  // uv[92] = 0.9867;
  // uv[93] = 0.9867;
  // uv[94] = 0.9867;
  // uv[95] = 0.986699999999968;
  // uv[96] = 0.986699999999968;
  // uv[97] = 0.986699999999968;
  // uv[98] = 0.986699999999968;
  // uv[99] = 0.986699999999968;
  // uv[100] = 0.986699999999968;
  // uv[101] = 0.986699999999968;
  // uv[102] = 0.986699999999968;
  // uv[103] = 0.986699999999968;
  // uv[104] = 0.986699999999968;
  // uv[105] = 0.986699999999968;
  // uv[106] = 0.986699999999968;
  // uv[107] = 0.986699999999968;
  // uv[108] = 0.986699999999968;
  // uv[109] = 0.986699999999968;
  // uv[110] = 0.986699999999968;
  // uv[111] = 0.986699999999968;
  // uv[112] = 0.986699999999968;
  // uv[113] = 0.986699999999968;
  // uv[114] = 0.986699999999968;
  // uv[115] = 0.98669999999994;
  // uv[116] = 0.98669999999994;
  // uv[117] = 0.98669999999994;
  // uv[118] = 0.98669999999994;
  // uv[119] = 0.986699999999941;
  // uv[120] = 0.986699999999941;
  // uv[121] = 0.986699999999941;
  // uv[122] = 0.986699999999941;
  // uv[123] = 0.986699999999941;
  // uv[124] = 0.986699999999941;
  // uv[125] = 0.986699999849629;
  // uv[126] = 0.986699999850775;
  // uv[127] = 0.986699999851794;
  // uv[128] = 0.986699999852686;
  // uv[129] = 0.986699999853455;
  // uv[130] = 0.986699999854102;
  // uv[131] = 0.986699999854628;
  // uv[132] = 0.986699999855034;
  // uv[133] = 0.986699999855321;
  // uv[134] = 0.986699999855491;
  // uv[135] = 0.986698723548635;
  // uv[136] = 0.986698733277232;
  // uv[137] = 0.986698741918956;
  // uv[138] = 0.986698749493982;
  // uv[139] = 0.986698756018639;
  // uv[140] = 0.986698761506832;
  // uv[141] = 0.986698765970198;
  // uv[142] = 0.986698769418146;
  // uv[143] = 0.986698771857926;
  // uv[144] = 0.98669877329464;
  // uv[145] = 0.977101677061948;
  // uv[146] = 0.977174822334586;
  // uv[147] = 0.977239805587536;
  // uv[148] = 0.977296766424471;
  // uv[149] = 0.97734582889256;
  // uv[150] = 0.977387097618018;
  // uv[151] = 0.977420660116612;
  // uv[152] = 0.977446587125918;
  // uv[153] = 0.97746493316029;
  // uv[154] = 0.977475736600941;
  for (i=0; i<N1*(Nr1+1); i++){
    uv[count] = 0.424;
    count++;
  }
  for (i=0; i<N1; i++){
    uv[count] = 0.431;
    count++;
  }
  // uv[155] = 0.423999999984612;
  // uv[156] = 0.423999999984577;
  // uv[157] = 0.423999999984517;
  // uv[158] = 0.423999999984433;
  // uv[159] = 0.423999999984325;
  // uv[160] = 0.423999999984192;
  // uv[161] = 0.423999999984035;
  // uv[162] = 0.424000000000007;
  // uv[163] = 0.424000000000007;
  // uv[164] = 0.424000000000007;
  // uv[165] = 0.424000000000007;
  // uv[166] = 0.424000000000007;
  // uv[167] = 0.424000000000007;
  // uv[168] = 0.424000000000008;
  // uv[169] = 0.424000000046193;
  // uv[170] = 0.424000000046299;
  // uv[171] = 0.424000000046477;
  // uv[172] = 0.424000000046729;
  // uv[173] = 0.424000000047054;
  // uv[174] = 0.424000000047453;
  // uv[175] = 0.424000000047926;
  // uv[176] = 0.42400059003143;
  // uv[177] = 0.424000591383448;
  // uv[178] = 0.424000593664057;
  // uv[179] = 0.424000596876971;

  // uv[180] = 0.42400060102742;
  // uv[181] = 0.424000606122196;
  // uv[182] = 0.424000612170774;
  // uv[183] = 0.431303952047274;
  // uv[184] = 0.431320688303582;
  // uv[185] = 0.431348919346656;
  // uv[186] = 0.431388691134095;
  // uv[187] = 0.431440068445479;
  // uv[188] = 0.431503136449684;
  // uv[189] = 0.431578000570174;
  uv[count] = 4.2;
  count++;
  uv[count] = -17.1;
} else {
  for (int i=0; i<NEQ; i++){
    uv[i] = p[i+34];
  }
}

  // input idv
  count = 1;
  for (i=0; i<N1; i++){
    idv[count] = 1.0;
    count++;
  }
  count++;
  for (i=0; i<N2; i++){
    idv[count] = 1.0;
    count++;
  }
  count++;
  for (i=0; i<N3; i++){
    idv[count] = 1.0;
    count++;
  }
  count++;

  // skip lpot, spot,
  for (i=0; i<2*N1+N2+3*N3+8; i++){
    count++;
  }
  for (i=0; i<N3*Nr2; i++){
    idv[count] = 1.0;
    count++;
  }
  for (i=0; i<N3+N1; i++){
    count++;
  }
  for (i=0; i<N1*Nr1; i++){
    idv[count] = 1.0;
    count++;
  }


  // Call IDACreate and IDAInit to initialize IDA memory
  mem = IDACreate();
  if(check_flag((void *)mem, "IDACreate", 0)) return(1);

   retval = IDASetId(mem, id);
  if(check_flag(&retval, "IDASetId", 1)) return(1);

  retval = IDAInit(mem, resrob, t0, yy, yp);
  if(check_flag(&retval, "IDAInit", 1)) return(1);

  retval = IDASVtolerances(mem, rtol, avtol);
  if(check_flag(&retval, "IDASVtolerances", 1)) return(1);

  // Free avtol
  N_VDestroy_Serial(avtol);

  // Call IDARootInit to specify the root function grob with 2 components
  // activate when using stop condition //
  retval = IDARootInit(mem, 1, grob);
  if(check_flag(&retval, "IDARootInit", 1)) return(1);

  // Call IDADense and set up the linear solver.
  retval = IDADense(mem, NEQ);
  if(check_flag(&retval, "IDADense", 1)) return(1);

  //retval = IDADlsSetDenseJacFn(mem, jacrob);
  //if(check_flag(&retval, "IDADlsSetDenseJacFn", 1)) return(1);

  retval = IDACalcIC(mem, IDA_YA_YDP_INIT, tout1);
  if(check_flag(&retval, "IDACalcIC", 1)) return(1);

  retval = IDAGetConsistentIC(mem,yy,yp);
  if(check_flag(&retval, "IDAGetConsistentIC", 1)) return(1);

  //retval = IDASetInitStep(mem, 1e-3);
  //if(check_flag(&retval, "IDASetInitStep", 1)) return(1);

  //retval = IDASetMaxStep(mem, 0);
  //if(check_flag(&retval, "IDASetMaxStep", 1)) return(1);

  //retval = IDASetStopTime(mem, 30000.000000);
  //if(check_flag(&retval, "IDASetStopTime", 1)) return(1);

  // PrintOutput(mem,0,yy);

  // yprint = NV_DATA_S(yy);
  // printf("attempted consistent ics: \n");
  //  for (int i=0; i<NEQ; i++){
 	//   printf("%i ", i);
 	//   printf("%15.4e \n", yprint[i]);
  //
  //  }

  // In loop, call IDASolve, print results, and test for error.
  // Break out of loop when NOUT preset output times have been reached.
  iout = 0;
  tout = tout1;
  int index = 0;

  while(1) {

    retval = IDASolve(mem, tout, &tret, yy, yp, IDA_NORMAL);

    // PrintOutput(mem,tret,yy);

    yprint = NV_DATA_S(yy);

    if (retval == IDA_ROOT_RETURN) {
      retvalr = IDAGetRootInfo(mem, rootsfound);
      check_flag(&retvalr, "IDAGetRootInfo", 1);
    }

    if (cc==1){
      if (tret >= 200.0){
           // printf("%15.15e ", tret-200.0);
           output[index] = tret-200.0;
           index++;
           for(i=0; i<NEQ; i++){
             // printf("%15.15e ", yprint[i]);
             output[index] = yprint[i];
             index++;
           }
         }
       } else {
              output[index] = tret;
              index++;
              for(i=0; i<NEQ; i++){
                output[index] = yprint[i];
                index++;
              }
            }

    if( retval == IDA_ROOT_RETURN )
    break;

    if(check_flag(&retval, "IDASolve", 1)) return(1);

    if (retval == IDA_SUCCESS) {
      iout++;
      tout += dt;
    }

    if (iout == NOUT)
    break;

	if (tret >= tfinal)
    break;
    //if (tret == 30000.000000) break;
   }
  // PrintOutput(mem,tret,yy);


  // Free memory
  IDAFree(&mem);
  N_VDestroy_Serial(yy);
  N_VDestroy_Serial(yp);

  return(0);
}

double Up1(double theta){
  return (-0.5057e1 - 0.136e2 * pow(theta , 2) + 0.1216e3 * pow(theta , 4) - 0.1851e3 * pow(theta , 6) - 0.4543e2 * pow(theta , 8) + 0.1277e3 * pow(theta , 10)) / (-1 - 0.504e1 * pow(theta , 2) + 0.323e2 * pow(theta , 4) - 0.4095e2 * pow(theta , 6) - 0.2594e2 * pow(theta , 8) + 0.4063e2 * pow(theta , 10));
}


double Un1(double theta){
  return (-0.5e-1 + 0.2325e1 * exp(-0.100e3 * pow(theta , 0.115e1)) - 0.1721e0 * tanh(0.20000000e2 * theta - 0.21000000e2) - 0.25e-2 * tanh(0.39347962e2 * theta - 0.37241379e2) - 0.34e-1 * tanh(0.89411765e2 * theta - 0.60294118e1) - 0.2e-2 * tanh(0.70422535e1 * theta - 0.13661972e1) - 0.155e-1 * tanh(0.77519380e2 * theta - 0.81395349e1));
}


// Define the system residual function.
int resrob(realtype tres, N_Vector yy, N_Vector yp, N_Vector rr, void *user_data)
{
  realtype *uv, *upv, *resv;
  double Dbulk = p[0];
   double Dsn = p[1];
   double Dsp = p[2];
   double Rpn = p[3];
   double Rpp = p[4];
   // double Tr = p[6];
   double brugn = p[6];
   double brugp = p[7];
   double brugs = p[8];
   double c0 = p[9];
   double ctn = p[10];
   double ctp = p[11];
   double efn = p[12];
   double efp = p[13];
   double en = p[14];
   double ep = p[15];
   double es = p[16];
   // double ratetest = p[18]; //# needs to be replaced with ratetest
   double kn = p[17];
   double kp = p[18];
   double ln1 = p[19];
   double lp = p[20];
   double ls = p[21];
   double sigman = p[22]*(1-en-efn);
   double sigmap = p[23]*(1-ep-efp);
   double t1 = p[24];
   double ratetest = p[30];

  // static pars (for now)
   double mu = 1E-4;
   // double  F = 96487;
   double Kappa = 0.1170027489e1;
   // double R = 0.83143e1;
   double i_applied = 0.171e2 * ratetest;


   // double iapp = 0.171e2 * ratetest;
   // double  socn = 0.9867e0;
   // double  socp = 0.424e0;


     uv = NV_DATA_S(yy);
    upv = NV_DATA_S(yp);
   resv = NV_DATA_S(rr);

   double inittime = 200;
   double ff = 1;
   if (cc == 1){
     ff = (tanh(100*((tres+100)-(inittime+100)))+tanh(100*(tres+100)))/2;
   }
   // double ff = (tanh(100*((tres+100)-(inittime+100)))+tanh(100*(tres+100)))/2;
   double iapp = uv[NEQ-1];

   //# coefficients
   double ap=(3/Rpp)*(1-ep-efp);
   double an=(3/Rpn)*(1-en-efn);

   //# therse might be needed
   double Keffn = Kappa*pow(en,brugn);
   double Keffs = Kappa*pow(es,brugs);
   double Keffp = Kappa*pow(ep,brugp);
   double D2pos = pow(ep,brugp)*Dbulk;
   double D2sep = pow(es,brugs)*Dbulk;
   double D2neg = pow(en,brugn)*Dbulk;


   double h = lp/(N1+1);
   double h2 = ls/(N2+1);
   double h3 = ln1/(N3+1);

   double hr1 = Rpp/(Nr1+1);
   double hr2 = Rpn/(Nr2+1);

   double jp[500];
   double jn[500];

   // input resv
   // pre-calculate some values to iterate over
   int start_lpot = N1+N2+N3+4;
   int start_spot = start_lpot+N1+N2+N3+4;
   int start_p_surface = NEQ-2-N1;
   int start_n_surface = start_spot+N1+N3+4+N3*(Nr2+1);
   int count = 0;
   int i=0;

   //
   for (i=0; i<N1; i++){
 	  jp[i] = 2.0*kp*pow(uv[i+1]*c0,0.5)*pow(-1.0*uv[start_p_surface+i]*ctp+ctp,0.5)*pow(uv[start_p_surface+i]*ctp,0.5)*sinh(0.1677008217E-2*F/R*(uv[start_spot+1+i]-1.0*uv[start_lpot+1+i]-1.0*Up1(uv[start_p_surface+i])));
   }
   //
   for (i=0; i<N3; i++){
 	  jn[i] = 2.0*kn*pow(uv[N1+N2+3+i]*c0,0.5)*pow(-1.0*uv[start_n_surface+i]*ctn+ctn,0.5)*pow(uv[start_n_surface+i]*ctn,0.5)*sinh(0.1677008217E-2*F/R*(uv[start_spot+N1+2+i]-1.0*uv[start_lpot+N1+N2+3+i]-1.0*Un1(uv[start_n_surface+i])));
   }




      // input resv

      resv[0] = 0.5*D2pos*(-1.0*uv[2]-3.0*uv[0]+4.0*uv[1])/h;
      count ++;
      for (i=0; i<N1; i++){
        // -1
        resv[count] = (D2pos/(h*h)*(uv[count-1]-2.0*uv[count]+uv[count+1])+ap*(1.0-1.0*t1)*jp[i]/c0)*ff-ep*upv[count];
        count++;
      }
      // +97
      resv[count] = 0.5*D2pos*(uv[N1-1]+3.0*uv[N1+1]-4.0*uv[N1])/h-0.5*D2sep*(-1.0*uv[N1+3]-3.0*uv[N1+1]+4.0*uv[N1+2])/h2;
      count++;
      for (i=0; i<N2; i++){
        // -2
        resv[count] = D2sep/(h2*h2)*(uv[count-1]-2.0*uv[count]+uv[count+1])*ff-es*upv[count];
        count++;
      }
      //+91
      resv[count] = 0.5*D2sep*(uv[count-2]+3.0*uv[count]-4.0*uv[count-1])/h2-0.5*D2neg*(-1.0*uv[count+2]-3.0*uv[count]+4.0*uv[count+1])/h3;
      count++;
      for (i=0; i<N3; i++){
        // -3
        resv[count] = (D2neg/(h3*h3)*(uv[count-1]-2.0*uv[count]+uv[count+1])+an*(1.0-1.0*t1)*jn[i]/c0)*ff-en*upv[count];
        count++;
      }
      //+81
      resv[count] = 0.5*D2neg*(uv[count-2]+3.0*uv[count]-4.0*uv[count-1])/h3;
      count++;



      // lpot
      // extra bc
      //+81
      resv[count] = 0.5*(-1.0*uv[count+2]-3.0*uv[count]+4.0*uv[count+1])/h;
      count++;
      for (int i=0; i<N1; i++){
           resv[count] = -0.5*sigmap/h*(uv[start_spot+2+i]-1.0*uv[start_spot+i])-0.5*Keffp/h*(uv[start_lpot+2+i]-1.0*uv[start_lpot+i])+0.29815E3*Keffp*R/F*(1.0-1.0*t1)/h*(uv[i+2]-1.0*uv[i])/uv[i+1]-iapp;
           count++;
         }
      resv[count] = 0.5*Keffp*(uv[count-2]+3.0*uv[count]-4.0*uv[count-1])/h-0.5*Keffs*(-1.0*uv[count+2]-3.0*uv[count]+4.0*uv[count+1])/h2;
      count++;
      // liquid pot in separator
       for (int i=0; i<N2; i++){
         resv[count] = -0.5*Keffs/h2*(uv[start_lpot+N1+3+i]-1.0*uv[start_lpot+N1+1+i])+0.29815E3*Keffs*R/F*(1.0-1.0*t1)/h2*(uv[N1+2+i]-1.0*uv[N1+1+i])/uv[N1+2+i]-iapp;
         count++;
       }
       resv[count] = 0.5*Keffs*(uv[start_lpot+N1+N2]+3.0*uv[start_lpot+N1+N2+2]-4.0*uv[start_lpot+N1+N2+1])/h2-0.5*Keffn*(-1.0*uv[start_lpot+N1+N2+4]-3.0*uv[start_lpot+N1+N2+2]+4.0*uv[start_lpot+N1+N2+3])/h3;
        count++;
        // neg liquid potential
        for (int i=0; i<N3; i++){
          resv[count] = -0.5*sigman/h3*(uv[start_spot+N1+4+i]-1.0*uv[start_spot+N1+2+i])-0.5*Keffn/h3*(uv[start_lpot+N1+N2+4+i]-1.0*uv[start_lpot+N1+N2+2+i])+0.29815E3*Keffn*R/F*(1.0-1.0*t1)/h3*(uv[N1+N2+4+i]-1.0*uv[N1+N2+2+i])/uv[N1+N2+3+i]-iapp;
          count++;
        }
        resv[count] = uv[start_spot-1];
        count++;


      // spot
      //+81
      resv[count] = 0.5*(-1.0*uv[start_spot+2]-3.0*uv[start_spot]+4.0*uv[start_spot+1])/h+1.0*iapp/sigmap;
      count++;
      for (i=0; i<N1; i++){
        resv[count] = sigmap/(h*h)*(uv[count-1]-2.0*uv[count]+uv[count+1])-ap*F*jp[i];
        count++;
      }
      resv[count] = 0.5*(uv[count-2]+3.0*uv[count]-4.0*uv[count-1])/h;
      count++;

      resv[count] = 0.5*(-1.0*uv[count+2]-3.0*uv[count]+4.0*uv[count+1])/h3;
      count++;

      for (i=0; i<N3; i++){
        resv[count] = sigman/(h3*h3)*(uv[count-1]-2.0*uv[count]+uv[count+1])-an*F*jn[i];
        count++;
      }

      resv[count] = 0.5*(uv[count-2]+3.0*uv[count]-4.0*uv[count-1])/h3+1.0*iapp/sigman;
      count++;



      // positive particle -
      // center BC
      // for (int j=0; j<Nr)

      // negative particle - center
      // n center
      //+95
      for (i=0; i<N3; i++){
        resv[count] = -0.5*(-1.0*uv[count+2*N3]-3.0*uv[count]+4.0*uv[count+N3])/hr2;
        count++;
      }


      // negative particle - internal points
      for (double j=1.0; j<Nr2+1.0; j++){
        for (i=0; i<N3; i++){
          // double c = (double) j;
          //-41
          resv[count] = 1.0/(j*j)*Dsn*(j*(uv[count+N3]-uv[count-N3])+(j*j)*(uv[count+N3]-2*uv[count]+uv[count-N3]))/(hr2*hr2)*ff-upv[count];
          count++;
        }
      }

      // negative particle - surface
      //+35
      for (i=0; i<N3; i++){
        resv[count] = 0.5*Dsn*(uv[count-2*N3]+3.0*uv[count]-4.0*uv[count-N3])/hr2+1.0*jn[i]/ctn;
        count++;
      }

      // positive particle - center
      //+1
      for (i=0; i<N1; i++){
        resv[count] = -0.5*(-1.0*uv[count+2*N1]-3.0*uv[count]+4.0*uv[count+N1])/hr1;
        count++;
      }

      // positive particle - internal points
      //-139
      for (double j=1.0; j<Nr1+1.0; j++){
        for (i=0; i<N1; i++){
          // double c = (double) j;
          resv[count] = 1.0/(j*j)*Dsp*(j*(uv[count+N1]-uv[count-N1])+(j*j)*(uv[count+N1]-2*uv[count]+uv[count-N1]))/(hr1*hr1)*ff-upv[count];
          count++;
        }
      }

      // p surface
      //-20
      for (i=0; i<N1; i++){
        resv[count] =  0.5*Dsp*(uv[count-2*N1]+3.0*uv[count]-4.0*uv[count-N1])/hr1+1.0*jp[i]/ctp;
        count++;
      }




      resv[NEQ-2] = uv[NEQ-2]-(uv[start_spot]-uv[start_spot+N1+N3+3]);
         // count++;
         // resv[130] = uv[130]-(uv[42]-uv[59]);
         // enforce either current = applied current or voltage = 4.2
         if (cc == 1){
           resv[NEQ-1] = iapp-i_applied;
         } else {
           resv[NEQ-1] = uv[NEQ-2]-4.2;
         }


  return(0);
}

// activate when using stop condition //
static int grob(realtype t, N_Vector yy, N_Vector yp, realtype *gout,
                void *user_data)
{
realtype *yval, y1;

   yval = NV_DATA_S(yy);
    if (cc == 1){
        if (yval[NEQ-1] > 0){
            gout[0] = (yval[NEQ-2]-4.2);
        } else {
             gout[0] = (yval[NEQ-2]-2.8);
        }

    } else {
        gout[0] = yval[NEQ-1] - 0.01*17.1;
    }
   return(0);
}

//Print Output
 static void PrintOutput(void *mem, realtype t, N_Vector y)
{
  realtype *yval;
  int retval, kused;
  long int nst, i;
  realtype hused;

  yval  = NV_DATA_S(y);

  retval = IDAGetLastOrder(mem, &kused);
  check_flag(&retval, "IDAGetLastOrder", 1);
  retval = IDAGetNumSteps(mem, &nst);
  check_flag(&retval, "IDAGetNumSteps", 1);
  retval = IDAGetLastStep(mem, &hused);
  check_flag(&retval, "IDAGetLastStep", 1);
#if defined(SUNDIALS_EXTENDED_PRECISION)
  printf("%10.4Le ",t);
  for (i = 0; i < NEQ; i++) { printf("%12.4Le ",yval[i]); }
  printf("\n");
#elif defined(SUNDIALS_DOUBLE_PRECISION)
  printf("%10.4le ",t);
  for (i = 0; i < NEQ; i++) { printf("%12.4le ",yval[i]); }
  printf("\n");
#else
  printf("%10.4e ",t);
  for (i = 0; i < NEQ; i++) { printf("%12.4e ",yval[i]); }
  printf("\n");
#endif
}

static int check_flag(void *flagvalue, char *funcname, int opt)
{
  int *errflag;
  // Check if SUNDIALS function returned NULL pointer - no memory allocated
  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr,
            "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return(1);
  } else if (opt == 1) {
    // Check if flag < 0
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      fprintf(stderr,
              "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
              funcname, *errflag);
      return(1);
    }
  } else if (opt == 2 && flagvalue == NULL) {
    // Check if function returned NULL pointer - no memory allocated
    fprintf(stderr,
            "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return(1);
  }

  return(0);
}
