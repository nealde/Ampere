
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../../../ida/ida.h"
#include "../../../ida/ida_dense.h"
#include "../../../ida/nvector_serial.h"
#include "../../../ida/sundials_math.h"
#include "../../../ida/sundials_types.h"


// Problem Constants
#define NEQ   7
#define NOUT  30000

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

/*
//--------------------------------------------------------------------
// Main Program

Parabolic approximation to the Single Particle Model - a parabola is used to approximate the concentration
in the particles.

input holds (up to) 14 model parameters, a continuation flag, final time, and initial conditions.
The model input parameters are in the following order: [0:16]
Dn, Dp, Rn, Rp, T, an, ap, ce, csnmax, cspmax, kn, kp, ln, lp, iapp, cc

A continuation flag (initial) is given to determine if the model will use the initial conditions here
or will expect values to be given. [16]
initial = 1 means the model ICs are supplied by this code
initial = 0 means that input contains the ICs. This is used for chaining together charges / discharges.

t_final (tf) is given, to simulate time-restricted charging or discharging. [17]

The next 8 parameters are the initial conditions (optional) [18:26]

//--------------------------------------------------------------------
*/

int cc = 1;
double p[100];

int main(double* input, double* output, int n)
{
  void     *mem;
  N_Vector  yy, yp, avtol, id;
  realtype  rtol, *uv, *upv, *atval, *idv, *yprint;
  realtype  dt, t0, tout1, tout, tret;
  realtype  MapleGenVar1, MapleGenVar2;
  int       iout, retval, retvalr;
  int       rootsfound[1];
  int       step,count;
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

  for (i=0; i<n; i++){
    p[i] = input[i];
  }

// Integration limits
    t0 = 0.000000;
    dt = 2.0;
 tout1 = 2.0;
  double tfinal = dt*NOUT;
  if (p[17] > 0){
      tfinal = p[17]+200.0;
  }
    cc = p[15];
    int initial = (int) p[16];
        if (initial == 1){
      // input uv
      uv[0] = 0.49503111E5;
      uv[1] = 0.30555E3;
      uv[2] = 0.49503111E5;
      uv[3] = 0.30555E3;
      uv[4] = 0.3758381197E1;
      uv[5] = 0.9691367515;
      uv[6] = p[14];

      // input upv
      upv[0] = -0.932768144931441E-3*uv[6]/p[6]/p[13]/p[3]*(tanh(100.0*t0-20000.0)/2.0+tanh(100.0*t0+10000.0)/2.0);
      upv[1] = 0.932768144931441E-3*uv[6]/p[5]/p[12]/p[2]*(tanh(100.0*t0-20000.0)/2.0+tanh(100.0*t0+10000.0)/2.0);
        } else {
            for (i=0; i<NEQ; i++){
                uv[i] = p[i+18];
//                printf("%lf ", uv[i]);
            }
        }
      // input idv
      idv[0] = 1.0;
      idv[1] = 1.0;
      idv[2] = 1.0;
      idv[3] = 1.0;
      idv[4] = 1.0;
      idv[5] = 1.0;



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

  //retval = IDASetStopTime(mem, 10000.000000);
  //if(check_flag(&retval, "IDASetStopTime", 1)) return(1);

  // PrintOutput(mem,0,yy);

  yprint = NV_DATA_S(yy);

  // endt = clock();

//  printf("%15.4e ", t0*dt);
//
//  for(i=0; i < NEQ-1; i++)
//     {
//      printf("%15.4e ", yprint[i]);
//     }
//    printf("%15.15e ", yprint[66]); // have to report current scaled by curr density
//  printf("\n");


  // In loop, call IDASolve, print results, and test for error.
  // Break out of loop when NOUT preset output times have been reached.
  iout = 0;
  tout = tout1;
  int index = 0;
//  int count1=0;
  while(1) {

    retval = IDASolve(mem, tout, &tret, yy, yp, IDA_NORMAL);
//      printf("%d ", count1);
//      count1 += 1;
    // PrintOutput(mem,tret,yy);

    yprint = NV_DATA_S(yy);

    if (retval == IDA_ROOT_RETURN) {
      retvalr = IDAGetRootInfo(mem, rootsfound);
      check_flag(&retvalr, "IDAGetRootInfo", 1);
    }
    // print to console
//       if (tret >= 200.){
//       printf("%15.15e ", tret-200.);
//       for(i=0; i < NEQ; i++)
        // { printf("%15.15e ", yprint[i]);}
// //      printf("%15.15e ", yprint[6]);
//       printf("\n");
//       }
      if (tret >= 200.){
        // printf("%15.15e ", tret-200.);
        output[index] = tret-200.;
        index++;
        for(i=0; i<NEQ; i++){
        //   printf("%15.15e ", yprint[i]);
          output[index] = yprint[i];
          index++;
        }
        // printf("\n");
      }
//    fprintf(savetime,"%15.15e\n", tret);
//    for(i=0; i < NEQ; i++)
//    { fprintf(savefile,"%15.15e\t", yprint[i]);}
//    fprintf(savefile,"\n");
//
//    if(tout == NOUT*dt || retval == IDA_ROOT_RETURN)
//    {for(i=0; i < NEQ-1; i++)
//    {fprintf(ic_out,"%e\n",yprint[i]);}
//    for(i=0; i<1; i++)
//    {fprintf(ic_out,"%e\n", yprint[NEQ-1]=0);}}

    if( retval == IDA_ROOT_RETURN ){
//        printf("%15.15e ", tret-200.);
//      for(i=0; i < NEQ; i++)
//        { printf("%15.15e ", yprint[i]);}
////      printf("%15.15e ", yprint[6]);
//      printf("\n");
//        printf("Root found");
        break;
    }
//    break;

    if(check_flag(&retval, "IDASolve", 1)) return(1);

    if (retval == IDA_SUCCESS) {
      iout++;
      tout += dt;
    }

    if (iout == NOUT)
    break;

    if (tret >= tfinal)
    break;
    //if (tret == 10000.000000) break;
   }
  // PrintOutput(mem,tret,yy);


//  filetime = fopen("caltime.txt","w");
//  fprintf(filetime,"%lf", (double) (endt - startt) / CLOCKS_PER_SEC);
//   printf("\n calculation time : %lf [sec] \n", (double) (endt - startt) / CLOCKS_PER_SEC);

//  fclose(savefile);
//  fclose(savetime);
//  fclose(filetime);

  // Free memory
  IDAFree(&mem);
  N_VDestroy_Serial(yy);
  N_VDestroy_Serial(yp);

  return(0);
}

// Define the system residual function.
int resrob(realtype tres, N_Vector yy, N_Vector yp, N_Vector rr, void *user_data)
{
  realtype *uv, *upv, *resv;
  realtype  MapleGenVar1, MapleGenVar2, MapleGenVar3, MapleGenVar4, MapleGenVar5;
  realtype  MapleGenVar6, MapleGenVar7, MapleGenVar8, MapleGenVar9, MapleGenVar10;
  realtype  MapleGenVar11, MapleGenVar12, MapleGenVar13, MapleGenVar14, MapleGenVar15;

    uv = NV_DATA_S(yy);
   upv = NV_DATA_S(yp);
  resv = NV_DATA_S(rr);

      // input resv
      resv[0] = -0.932768144931441E-3*uv[6]/p[6]/p[13]/p[3]*(tanh(100.0*tres-20000.0)/2.0+tanh(100.0*tres+10000.0)/2.0)-upv[0];
       resv[1] = 0.932768144931441E-3*uv[6]/p[5]/p[12]/p[2]*(tanh(100.0*tres-20000.0)/2.0+tanh(100.0*tres+10000.0)/2.0)-upv[1];
      resv[2] = -5.0*p[1]*(uv[2]-1.0*uv[0])/p[3]-30.0/96487.0*uv[6]/p[6]/p[13]-0.5E-3*p[1]*(upv[2]-1.0*upv[0])/p[3];
      resv[3] = -5.0*p[0]*(uv[3]-1.0*uv[1])/p[2]+0.310922714977147E-3*uv[6]/p[5]/p[12]-0.5E-3*p[0]*(upv[3]-1.0*upv[1])/p[2];
      MapleGenVar1 = -2.0*p[11]*pow(p[7],0.5)*pow(p[9]-1.0*uv[2],0.5)*pow(uv[2]
,0.5)*sinh(0.58024728479848E4/p[4]*(uv[4]-1.0*(-0.4656E1+0.88669E2*uv[2]*uv[2]/
(p[9]*p[9])-0.401119E3*uv[2]*uv[2]*uv[2]*uv[2]/(p[9]*p[9]*p[9]*p[9])+0.342909E3
*uv[2]*uv[2]*uv[2]*uv[2]*uv[2]*uv[2]/(p[9]*p[9]*p[9]*p[9]*p[9]*p[9])-0.462471E3
*uv[2]*uv[2]*uv[2]*uv[2]*uv[2]*uv[2]*uv[2]*uv[2]/(p[9]*p[9]*p[9]*p[9]*p[9]*p[9]
*p[9]*p[9])+0.433434E3*pow(uv[2],10.0)/pow(p[9],10.0))/(-0.1E1+0.18933E2*uv[2]*
uv[2]/(p[9]*p[9])-0.79532E2*uv[2]*uv[2]*uv[2]*uv[2]/(p[9]*p[9]*p[9]*p[9])+
0.37311E2*uv[2]*uv[2]*uv[2]*uv[2]*uv[2]*uv[2]/(p[9]*p[9]*p[9]*p[9]*p[9]*p[9])
-0.73083E2*uv[2]*uv[2]*uv[2]*uv[2]*uv[2]*uv[2]*uv[2]*uv[2]/(p[9]*p[9]*p[9]*p[9]
*p[9]*p[9]*p[9]*p[9])+0.9596E2*pow(uv[2],10.0)/pow(p[9],10.0))))+
0.310922714977147E-3*uv[6]/p[6]/p[13];
      MapleGenVar2 = MapleGenVar1+0.222492945955597E-1*p[11]*pow(p[7],0.5)/pow(
p[9]-0.49503111E5,0.5)*sinh(0.58024728479848E4/p[4]*(0.3758381197E1-1.0*(
-0.4656E1+0.217288527184808E12/(p[9]*p[9])-0.240881365936549E22/(p[9]*p[9]*p[9]
*p[9])+0.504630900734571E31/(p[9]*p[9]*p[9]*p[9]*p[9]*p[9])
-0.166780183522358E41/(p[9]*p[9]*p[9]*p[9]*p[9]*p[9]*p[9]*p[9])+
0.383043329808881E50/pow(p[9],10.0))/(-0.1E1+0.463964145889766E11/(p[9]*p[9])
-0.477608310642617E21/(p[9]*p[9]*p[9]*p[9])+0.549075222210779E30/(p[9]*p[9]*p
[9]*p[9]*p[9]*p[9])-0.263558064232449E40/(p[9]*p[9]*p[9]*p[9]*p[9]*p[9]*p[9]*p
[9])+0.848037715741272E49/pow(p[9],10.0))))*upv[2];
      MapleGenVar3 = MapleGenVar2;
      MapleGenVar5 = -0.449452451494607E-6*p[11]*pow(p[7],0.5)*pow(p[9]
-0.49503111E5,0.5)*sinh(0.58024728479848E4/p[4]*(0.3758381197E1-1.0*(-0.4656E1+
0.217288527184808E12/(p[9]*p[9])-0.240881365936549E22/(p[9]*p[9]*p[9]*p[9])+
0.504630900734571E31/(p[9]*p[9]*p[9]*p[9]*p[9]*p[9])-0.166780183522358E41/(p[9]
*p[9]*p[9]*p[9]*p[9]*p[9]*p[9]*p[9])+0.383043329808881E50/pow(p[9],10.0))/(
-0.1E1+0.463964145889766E11/(p[9]*p[9])-0.477608310642617E21/(p[9]*p[9]*p[9]*p
[9])+0.549075222210779E30/(p[9]*p[9]*p[9]*p[9]*p[9]*p[9])-0.263558064232449E40/
(p[9]*p[9]*p[9]*p[9]*p[9]*p[9]*p[9]*p[9])+0.848037715741272E49/pow(p[9],10.0)))
)*upv[2];
      MapleGenVar7 = -0.2582018555551E3*p[11]*pow(p[7],0.5);
      MapleGenVar9 = pow(p[9]-0.49503111E5,0.5)/p[4];
      MapleGenVar12 = upv[4];
      MapleGenVar14 = -1.0*(0.8778782698518E7/(p[9]*p[9])*upv[2]
-0.194639376047739E18/(p[9]*p[9]*p[9]*p[9])*upv[2]+0.611635378715373E27/(p[9]*p
[9]*p[9]*p[9]*p[9]*p[9])*upv[2]-0.269526791594747E37/(p[9]*p[9]*p[9]*p[9]*p[9]*
p[9]*p[9]*p[9])*upv[2]+0.773776278038125E46/pow(p[9],10.0)*upv[2])/(-0.1E1+
0.463964145889766E11/(p[9]*p[9])-0.477608310642617E21/(p[9]*p[9]*p[9]*p[9])+
0.549075222210779E30/(p[9]*p[9]*p[9]*p[9]*p[9]*p[9])-0.263558064232449E40/(p[9]
*p[9]*p[9]*p[9]*p[9]*p[9]*p[9]*p[9])+0.848037715741272E49/pow(p[9],10.0));
      MapleGenVar15 = 1.0*(-0.4656E1+0.217288527184808E12/(p[9]*p[9])
-0.240881365936549E22/(p[9]*p[9]*p[9]*p[9])+0.504630900734571E31/(p[9]*p[9]*p
[9]*p[9]*p[9]*p[9])-0.166780183522358E41/(p[9]*p[9]*p[9]*p[9]*p[9]*p[9]*p[9]*p
[9])+0.383043329808881E50/pow(p[9],10.0))/pow(-0.1E1+0.463964145889766E11/(p[9]
*p[9])-0.477608310642617E21/(p[9]*p[9]*p[9]*p[9])+0.549075222210779E30/(p[9]*p
[9]*p[9]*p[9]*p[9]*p[9])-0.263558064232449E40/(p[9]*p[9]*p[9]*p[9]*p[9]*p[9]*p
[9]*p[9])+0.848037715741272E49/pow(p[9],10.0),2.0)*(0.1874484801126E7/(p[9]*p
[9])*upv[2]-0.385921855006339E17/(p[9]*p[9]*p[9]*p[9])*upv[2]+
0.665503897980201E26/(p[9]*p[9]*p[9]*p[9]*p[9]*p[9])*upv[2]
-0.425925658260062E36/(p[9]*p[9]*p[9]*p[9]*p[9]*p[9]*p[9]*p[9])*upv[2]+
0.171309984081864E46/pow(p[9],10.0)*upv[2]);
      MapleGenVar13 = MapleGenVar14+MapleGenVar15;
      MapleGenVar11 = MapleGenVar12+MapleGenVar13;
      MapleGenVar12 = cosh(0.58024728479848E4/p[4]*(0.3758381197E1-1.0*(
-0.4656E1+0.217288527184808E12/(p[9]*p[9])-0.240881365936549E22/(p[9]*p[9]*p[9]
*p[9])+0.504630900734571E31/(p[9]*p[9]*p[9]*p[9]*p[9]*p[9])
-0.166780183522358E41/(p[9]*p[9]*p[9]*p[9]*p[9]*p[9]*p[9]*p[9])+
0.383043329808881E50/pow(p[9],10.0))/(-0.1E1+0.463964145889766E11/(p[9]*p[9])
-0.477608310642617E21/(p[9]*p[9]*p[9]*p[9])+0.549075222210779E30/(p[9]*p[9]*p
[9]*p[9]*p[9]*p[9])-0.263558064232449E40/(p[9]*p[9]*p[9]*p[9]*p[9]*p[9]*p[9]*p
[9])+0.848037715741272E49/pow(p[9],10.0))));
      MapleGenVar10 = MapleGenVar11*MapleGenVar12;
      MapleGenVar8 = MapleGenVar9*MapleGenVar10;
      MapleGenVar6 = MapleGenVar7*MapleGenVar8;
      MapleGenVar4 = MapleGenVar5+MapleGenVar6;
      resv[4] = MapleGenVar3+MapleGenVar4;
      MapleGenVar1 = -2.0*p[10]*pow(p[7],0.5)*pow(p[8]-1.0*uv[3],0.5)*pow(uv[3]
,0.5)*sinh(0.58024728479848E4/p[4]*(uv[5]-0.7222-0.1387*uv[3]/p[8]-0.29E-1*pow(
uv[3]/p[8],0.5)+0.172E-1/uv[3]*p[8]-0.19E-2/pow(uv[3]/p[8],0.15E1)-0.2808*exp(
0.9-15.0*uv[3]/p[8])+0.7984*exp(0.4465*uv[3]/p[8]-0.4108)))-30.0/96487.0*uv[6]/
p[5]/p[12];
      MapleGenVar2 = MapleGenVar1+0.174799885583487E-2*p[10]*pow(p[7],0.5)/pow(
p[8]-0.30555E3,0.5)*sinh(0.58024728479848E4/p[4]*(0.2469367515-0.42379785E2/p
[8]-0.506919668192112*pow(1/p[8],0.5)+0.562919325805924E-4*p[8]
-0.35573792610515E-6/pow(1/p[8],0.15E1)-0.2808*exp(0.9-0.458325E4/p[8])+0.7984*
exp(0.136428075E3/p[8]-0.4108)))*upv[3];
      MapleGenVar3 = MapleGenVar2;
      MapleGenVar5 = -0.572082754323307E-5*p[10]*pow(p[7],0.5)*pow(p[8]
-0.30555E3,0.5)*sinh(0.58024728479848E4/p[4]*(0.2469367515-0.42379785E2/p[8]
-0.506919668192112*pow(1/p[8],0.5)+0.562919325805924E-4*p[8]
-0.35573792610515E-6/pow(1/p[8],0.15E1)-0.2808*exp(0.9-0.458325E4/p[8])+0.7984*
exp(0.136428075E3/p[8]-0.4108)))*upv[3];
      MapleGenVar6 = -0.202854317985807E2*p[10]*pow(p[7],0.5)*pow(p[8]
-0.30555E3,0.5)/p[4]*(upv[5]-0.1387*upv[3]/p[8]-0.829519993768795E-3/pow(1/p[8]
,0.5)*upv[3]/p[8]-0.184231492654533E-6*p[8]*upv[3]+0.174638157145385E-8/pow(1/p
[8],0.25E1)*upv[3]/p[8]+0.4212E1*upv[3]/p[8]*exp(0.9-0.458325E4/p[8])+0.3564856
*upv[3]/p[8]*exp(0.136428075E3/p[8]-0.4108))*cosh(0.58024728479848E4/p[4]*(
0.2469367515-0.42379785E2/p[8]-0.506919668192112*pow(1/p[8],0.5)+
0.562919325805924E-4*p[8]-0.35573792610515E-6/pow(1/p[8],0.15E1)-0.2808*exp(0.9
-0.458325E4/p[8])+0.7984*exp(0.136428075E3/p[8]-0.4108)));
      MapleGenVar4 = MapleGenVar5+MapleGenVar6;
      resv[5] = MapleGenVar3+MapleGenVar4;
      if(cc == 1){
        resv[6] = uv[6]- p[14];
    } else {
        resv[6] = uv[4]-uv[5] - 4.2;
    }

  return(0);
}

// activate when using stop condition //
static int grob(realtype t, N_Vector yy, N_Vector yp, realtype *gout,
                void *user_data)
{
realtype *yval, y1;

   yval = NV_DATA_S(yy);
//    if ((int) p[24] == 1 ){
//        if (p[15] < 0.0){
//            gout[0] = (yval[4]-yval[5]-4.2);
//        } else {
//            gout[0] = (yval[4]-yval[5]-2.5);
//        }
//    } else {
//        gout[0] = (yval[6]/30.0+0.1);
//    }

    if (cc == 1){
        if (yval[6] > 0){
            gout[0] = (yval[4]-yval[5]-4.2);
        } else {
             gout[0] = (yval[4]-yval[5]-2.5);
        }

    } else {
        gout[0] = yval[6] - 0.01;
    }
   return(0);
}

//Print Output
//  static void PrintOutput(void *mem, realtype t, N_Vector y)
// {
//   realtype *yval;
//   int retval, kused;
//   long int nst, i;
//   realtype hused;
//
//   yval  = NV_DATA_S(y);
//
//   retval = IDAGetLastOrder(mem, &kused);
//   check_flag(&retval, "IDAGetLastOrder", 1);
//   retval = IDAGetNumSteps(mem, &nst);
//   check_flag(&retval, "IDAGetNumSteps", 1);
//   retval = IDAGetLastStep(mem, &hused);
//   check_flag(&retval, "IDAGetLastStep", 1);
// #if defined(SUNDIALS_EXTENDED_PRECISION)
//   printf("%10.4Le ",t);
//   for (i = 0; i < NEQ; i++) { printf("%12.4Le ",yval[i]); }
//   printf("\n");
// #elif defined(SUNDIALS_DOUBLE_PRECISION)
//   printf("%10.4le ",t);
//   for (i = 0; i < NEQ; i++) { printf("%12.4le ",yval[i]); }
//   printf("\n");
// #else
//   printf("%10.4e ",t);
//   for (i = 0; i < NEQ; i++) { printf("%12.4e ",yval[i]); }
//   printf("\n");
// #endif
// }

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
