/**
 * Updated version for integration with C or Python code.
 * Original header is provided below.
 */
/* =======================================================================
*		soscascade.c 	Main code for doing cochlear filter 
*				operations for the second order 
*				section (sos) model.  Assumes that 
*				the filter coefficients have been 
*				designed.  This implements a CASCADE
*				of second order sections.
*	
*		Written : 	January 7, 1992
*		by :		Daniel Naar	
*									
*		Changes	:	Cleaned up by Malcolm
*				June 11, 1993-March 8, 1994
*				November 11, 1997 malcolm@interval.com
*				September 17, 1998 malcolm@interval.com
*                               
*		(c) 1997 Interval Research Corporation
*		[output,state] = soscascade(input, Coeffs, states)
*
* ========================================================================*/
#include	<stdio.h>
#include 	<math.h>

int soscascade(double* signal, int nSamples, double* coeffs, double* state,
	       int nChannels, double* outputData)
{
  if (0 == nSamples)
    return -1;

  double *a0, *a1, *a2, *b1, *b2;
  a0 = coeffs;
  a1 = coeffs + nChannels;
  a2 = coeffs + 2 * nChannels;
  b1 = coeffs + 3 * nChannels;
  b2 = coeffs + 4 * nChannels;

  double *state1, *state2;
  state1 = state;
  state2 = state + nChannels;

  int i, n;
  double in, *output;

  for (n = 0; n < nSamples; n++) {
    in = signal[n];
    output = &outputData[n * nChannels];
    for (i = 0; i < nChannels; i++) {
      output[i] = a0[i] * in                     + state1[i];
      state1[i] = a1[i] * in - b1[i] * output[i] + state2[i];
      state2[i] = a2[i] * in - b2[i] * output[i];
      in = output[i];
    }
  }
  return 0;
}
