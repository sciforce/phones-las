/**
 * Updated version for integration with C or Python code.
 * Original header is provided below.
 */
/* =======================================================================
*		agc.c-	Main code for doing cochlear filter 
*			automatic gain control.
*	
*		Written : 	January 7, 1992
*		by :		Daniel Naar	
*									
*		Changes	:	Cleaned up by Malcolm
*				June 11, 1993
*				November 11, 1997 malcolm@interval.com
*				Massive cleanup by malcolm on 9/17/98
*		(c) 1997 Interval Research Corporation
*		[output] = agc(input, Coeffs, output, states)
* ========================================================================*/

/*
 *	Test this command by trying the following... the correct results are 
 *	shown below.  
 *
 *	>> agc(ones(1,20),[.5;.5])
 *
 *	ans =
 *	
 *	Columns 1 through 7
 *	
 *	  1.0000    0.1000    0.4500    0.2750    0.3625    0.3187    0.3406
 *	
 *	Columns 8 through 14
 *	
 *	  0.3297    0.3352    0.3324    0.3338    0.3331    0.3334    0.3333
 *	
 *	Columns 15 through 20
 *	
 *	  0.3334    0.3333    0.3333    0.3333    0.3333    0.3333
 *	
 *	Note, if you change the value of EPS (the time constant), then the 
 *	first few results will change, but the asymptote should be the same.
 *
 *	This new version of the AGC mex function makes the state variables
 *	explicit.  To make sure the state is handled correctly, compare the
 *	output of these two sequences of commands.
 *	>> agc(ones(1,20),[.5;.5])
 *	and
 *	>> [output s] = agc(ones(1,10),[.5;.5])
 *	>> agc(ones(1,10),[.5; .5], s)
 *
 *	To check the inverse AGC function do the following
 *	>> output = agc(ones(1,20),[.5;.5])
 *	>> inverseagc(output, [.5; .5])
 *	The answer should be all ones, identical to the input.
 */

#include	<stdio.h>
#include 	<math.h>
   
#define	EPS	(1e-1)

/**
 * Performs single step of AGC.
 * Params:
 *   input - array of nChannels elements.
 *   output - array of nChannels elements.
 *   state - array of nChannels elements.
 *   n - nChannels.
 */
void agc_step(double *input, double *output, double *state, double epsilon, 
              double target, int n) {
  int i;
  double f, StateLimit=1. - EPS;
  double OneMinusEpsOverThree = (1.0 - epsilon) / 3.0;
  double EpsOverTarget = epsilon / target;
  double prevState;

  prevState = state[0];
  for (i = 0; i < n - 1; i++) {
    output[i] = fabs(input[i] * (1.0 - state[i]));
    f = output[i] * EpsOverTarget + 
      OneMinusEpsOverThree* (prevState + state[i] + state[i+1]);
    if (f > StateLimit) f = StateLimit;
    prevState = state[i];
    state[i] = f;	
  }
  output[i] = fabs(input[i] * (1.0 - state[i]));
  f = output[i] * EpsOverTarget + 
    OneMinusEpsOverThree*(prevState + state[i] + state[i]);
  if (f > StateLimit) f = StateLimit;
  state[i] = f;
}

int agc(double* inputData, int nChannels, int nSamples, int nStages,
	double *agcParams, double* state, double *outputData) {
  if (0 == nChannels || 0 == nSamples)
    return -1;

  int i, j;
  for (j = 0; j < nSamples; j++) {
    agc_step(inputData + j * nChannels, outputData + j * nChannels,
	     state + 0 * nChannels,
	     (double)agcParams[1+0*2], (double)agcParams[0+0*2], 
	     (int)nChannels);
    /* Target; Epsilon */
    for (i = 1; i < nStages; i++) {
      agc_step(outputData + j * nChannels, outputData + j * nChannels,
	       state + i * nChannels,
	       (double)agcParams[1+i*2], 
	       (double)agcParams[0+i*2], 
	       (int)nChannels);
    }
  }
  return 0;
}
