
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
// #define NEQ   7
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
// Main Program - moved to C by Neal Dawson-Elli

input holds 28+ parameters in this order:

first 24 are model inputs [0:23]
Dp, Dn, cspmax, csnmax, lp, ln, Rpp, Rpn, T, ce, ap, an, M_sei, rho_sei, Kappa_sei, kp, kn, ksei,
theta_p0, theta_n0, signparam, ratetest, endvolt, cc,

next 2 are node spacing [23:25]
N1, N2

Then whether (optionally) new initial conditions and the trigger value to look for those new conditions. [25:25+N1+N2+15]
initial = 1, model is initialized by the values in this file
initial = 0, model takes given inputs.
Then, t_final (tf), the maximum simulation time.
Because node spacing can change from run to run, it expects Python to handle
re-spacing the initial conditions as necessary.

In other words:

Dp = p[0]
Dn = p[1]
cspmax = p[2]
csnmax = p[3]
lp = p[4]
lnn = p[5]
Rpp = p[6]
Rpn = p[7]
T = p[8]
ce = p[9]
ap = p[10]
an = p[11]
// M_sei = p[12]
// rho_sei = p[13]
// Kappa_sei = p[14]
kp = p[15] 12
kn = p[16] 13
// ksei = p[17]
N1 = p[18] 14
N2 = p[19] 15
ratetest = p[20] 16
cc = p[21] 17
initial = p[22] 18
tf = p[23] 19
uv[i+] = p[24+] 20 +

//--------------------------------------------------------------------
*/
int cc = 1;
int N1 = 5;
int N2 = 5;
int NEQ = 24;
// define model constants
double F = 96487.0;
double R = 8.3143;
double TC = 30.0;
// define parameter vector
double p[5000];
double csp[1500];
double csn[1500];

int drive(double* input, double* output, int n)
{
  void     *mem;
  N_Vector  yy, yp, avtol, id;
  realtype  rtol, *uv, *upv, *atval, *idv, *yprint;
  realtype  dt, t0, tout1, tout, tret;
  int       iout, retval, retvalr;
  int       rootsfound[1];
  int       step,count;
  // int       err = 0;
  long      int i;
  // double*   x_init = NULL;
  // double*   r      = NULL;
  // double*   tf     = NULL;
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
  }
  N1 = (int) p[14];
  N2 = (int) p[15];
  NEQ = N1+N2+8;

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



// Integration limits
    t0 = 0.000000;
    dt = 10.0;
 tout1 = 10.0;
 // tfinal is now
  double tfinal = dt*NOUT;
  if (p[19] > 0){
      tfinal = p[19]+200.0;
  }
    cc = p[17];
    int initial = (int) p[18];
        if (initial == 1){
      // input uv
      // input uv
      for (i=0; i<N1+2; i++){
        uv[i] = 49503.111;
      }
      for (i=N1+2; i<N1+2+N2+2; i++){
        uv[i] = 305.55;
      }
      uv[N1+N2+4] = 3.67873289259766;    //#phi_p
      uv[N1+N2+5] = .182763748093840;    //#phi_n
      // uv[N1+N2+6] = 30.0;                  //#iint
      // uv[N1+N2+7] = 0.0;                   //#isei
      // uv[N1+N2+8] = 1e-10;               //#delta_sei
      // uv[N1+N2+9] = 0.0;                  //#Q
      // uv[N1+N2+10] = 0.0;                  //#cm
      // uv[N1+N2+11] = 0.0;                  //#cf
      uv[N1+N2+6] = 3.0596914450382;  // #pot
      uv[N1+N2+7] = 30.0 ;  	  	  //#it

      // input upv
      // upv[0] = -0.932768144931441E-3*uv[6]/p[6]/p[13]/p[3]*(tanh(100.0*t0-20000.0)/2.0+tanh(100.0*t0+10000.0)/2.0);
      // upv[1] = 0.932768144931441E-3*uv[6]/p[5]/p[12]/p[2]*(tanh(100.0*t0-20000.0)/2.0+tanh(100.0*t0+10000.0)/2.0);
        } else {
            for (i=0; i<NEQ; i++){
                uv[i] = p[i+20];
//                printf("%lf ", uv[i]);
            }
        }
      // input idv
      for (i=0; i<NEQ; i++){
        idv[i] = 1.0;
      }
      idv[0] = 0.0;
      idv[N1+1] = 0.0;
      idv[N1+2] = 0.0;
      idv[N1+N2+3] = 0.0;
      idv[N1+N2+4] = 0.0;
      idv[N1+N2+5] = 0.0;
      idv[N1+N2+6] = 0.0;
      idv[N1+N2+7] = 0.0;
	  // idv[N1+N2+8] = 0.0;



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

//       }
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

    if( retval == IDA_ROOT_RETURN ){

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

    uv = NV_DATA_S(yy);
   upv = NV_DATA_S(yp);
  resv = NV_DATA_S(rr);
   int i=0;
      // input resv
    double Dp = p[0];
    double Dn = p[1];
    double cspmax = p[2];
    double csnmax = p[3];
    double lp = p[4];
    double lnn = p[5];
    double Rpp = p[6];
    double Rpn = p[7];
    double T = p[8];
    double ce = p[9];
    double ap = p[10];
    double an = p[11];
    // double M_sei = p[12];
    // double rho_sei = p[13];
    // double Kappa_sei = p[14];
    double kp = p[12];
    double kn = p[13];
    // double ksei = p[17];
	// ksei=0.0;

    // double theta_p0 = p[18];
    // double theta_n0 = p[19];
	double signparam = 1.0;

    // double signparam = p[20];
    double ratetest = p[16];

	if (ratetest < 0){
		signparam = -1.0;
	}

    // double endvolt1 = p[22];
    // double cc = p[23];

//    double csp[N1+2];
//    double csn[N2+2];
//    double csp[150];
//    double csn[150];

    // csp = np.zeros(N1+2)
    for (i=0; i<N1+2; i++){
      csp[i] = fmax(1., fmin(uv[i], cspmax-1));
    }

    for (i=0; i<N2+2; i++){
      csn[i] = fmax(1., fmin(uv[N1+2+i], csnmax-1));
    }

// #     print(csp, csn)
// #     print(phi_p)
    double phi_p = uv[N1+N2+4];
    double phi_n = uv[N1+N2+5];
    // double iint = uv[N1+N2+6];
    // double isei = uv[N1+N2+7];
	// isei=0;
    // double delta_sei = uv[N1+N2+8];
    // double Q = uv[N1+N2+9];
    // double cm = uv[N1+N2+10];
    // double cf = uv[N1+N2+11];
    double pot = uv[N1+N2+6];
    double it = uv[N1+N2+7];
	double iint = it;
    double h1=Rpp/(N1+1);
    double h2=Rpn/(N2+1);

    // from numpy import tanh, sinh, exp
    double inittime = 200;
    double ff = (tanh(100*((tres+100)-(inittime+100)))+tanh(100*(tres+100)))/2;
    // double ff = 1;

    // # Governing Equations
    // # Positive Electrode
    resv[0] = 4*csp[1] - 3*csp[0] - csp[2]; //# boundary condition
    for (i=1; i<N1+1; i++){
        resv[i] =  Dp/pow(h1,2) *((csp[i+1]  - csp[i-1])/i + csp[i+1] + csp[i-1] - 2*csp[i])*ff - upv[i]; //# finite difference
    }
    resv[N1+1] = (csp[N1-1] + 3*csp[N1+1] - 4*csp[N1]) + (2*h1)*iint/F/Dp/ap/lp; //# second boundary condition

    // # negative electrode
    resv[N1+2] = 4*csn[1] - 3*csn[0] - csn[2];
    for (i=1; i<N2+1; i++){
        resv[N1+2+i] =  Dn/pow(h2,2) *((csn[i+1]  - csn[i-1])/i + csn[i+1] + csn[i-1] - 2*csn[i])*ff - upv[N1+2+i];
      }
    resv[N1+N2+3] = (csn[N2-1] + 3*csn[N2+1] - 4*csn[N2]) - (2*h2)*iint/F/Dn/an/lnn;


    // # additional equations
    // # positive electrode
    double theta_p = csp[N1+1]/cspmax;
    double Up = (-4.656+88.669*pow(theta_p,2)-401.119*pow(theta_p,4)+342.909*pow(theta_p,6)-462.471*pow(theta_p,8)+433.434*pow(theta_p,10))/(-1.0+18.933*pow(theta_p,2)-79.532*pow(theta_p,4)+37.311*pow(theta_p,6)-73.083*pow(theta_p,8)+95.96*pow(theta_p,10));
    double jp = 2*kp*pow(ce,(0.5))*pow((cspmax-csp[N1+1]),(0.5))*pow(csp[N1+1],(0.5))*sinh(0.5*F/R/T*(phi_p-Up));
    resv[N1+N2+4] = jp - (it/ap/F/lp);


    // #Negative electrode
    double theta_n = csn[N2+1]/csnmax;
    double Un = 0.7222+0.1387*theta_n+0.029*pow(theta_n,(0.5))-0.0172/theta_n+0.0019/pow(theta_n,1.5)+0.2808*exp(0.9-15*theta_n)-0.7984*exp(0.4465*theta_n-0.4108);
	double jn = 2*kn*pow(ce,(0.5))*pow((csnmax-csn[N2+1]),(0.5))*pow(csn[N2+1],(0.5))*sinh(0.5*F/R/T*(phi_n-Un));
    resv[N1+N2+5] = jn + (iint/an/F/lnn);

    resv[N1+N2+6] = pot - phi_p + phi_n;

    if(cc == 1){
        resv[N1+N2+7] = it - TC*ratetest;
      } else {
        resv[N1+N2+7] = phi_p-phi_n-4.2;
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
  if (yval[N1+N2+7] > 0){
	  gout[0] = (yval[N1+N2+6]-4.2);
  } else {
	   gout[0] = (yval[N1+N2+6]-2.5);
  }

} else {
  gout[0] = yval[N1+N2+7] - 0.01;
}
return(0);
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
