/**
 * Updated version for integration with C or Python code.
 * Original header is provided below.
 */
/* =======================================================================
*		sosfilters.c 	Main code for filtering with second
*				order sections.  Either one input channel
*				and a bunch of filters, or a bunch of
*				input channels and one filter, or a bunch
*				of input data and filters.  This implements
*				a PARALLEL bank of second order sections.
*	
*		Written : 	June 17, 1993
*		by :		Malcolm Slaney
*		Based on:	Code by Dan Naar and Malcolm Slaney
*		Changes: 	November 11, 1997 malcolm@interval.com
*		(c) 1997 Interval Research Corporation
*		[output] = sosfilters(input, Coeffs, output, states)
* ========================================================================*/

/*
 *	Test this command by trying the following... the correct results are 
 *	shown below.  (The two filters are both simple exponentials.)
 *	This first example is one input applied to a bunch of filters.
 *	>> sosfilters([1 0 0 0 0 0], [1 0 0 -.9 0; 1 0 0 -.8 0])
 *
 *	ans =
 *
 *	    1.0000    0.9000    0.8100    0.7290    0.6561    0.5905
 *	    1.0000    0.8000    0.6400    0.5120    0.4096    0.3277
 *
 *	This command can also filter an array of inputs.
 *	>> sosfilters([1 0 0 0 0 0;2 0 0 0 0 0], [1 0 0 -.9 0; 1 0 0 -.8 0])
 *
 *	ans =
 *
 *	    1.0000    0.9000    0.8100    0.7290    0.6561    0.5905
 *	    2.0000    1.6000    1.2800    1.0240    0.8192    0.6554
 *
 *	Or a bunch of inputs, independently applied to just one filter.
 *	>> sosfilters([1 0 0 0 0 0;2 0 0 0 0 0], [1 0 0 -.9 0])
 *
 *	ans =
 *
 *	    1.0000    0.9000    0.8100    0.7290    0.6561    0.5905
 *	    2.0000    1.8000    1.6200    1.4580    1.3122    1.1810
 *
 *	To check that the state variables are handled correctly, compare
 *	>> sosfilters([1 zeros(1,9)], [1 0 0 -.9 0; 1 0 0 -.8 0])
 *	and
 *	>> [output,s] = sosfilters([1 zeros(1,4)], [1 0 0 -.9 0; 1 0 0 -.8 0])
 *	>> sosfilters(zeros(1,5), [1 0 0 -.9 0; 1 0 0 -.8 0], s)
 */

#include	<stdio.h>
#include 	<math.h>

#define max(a,b) ((a) > (b)? (a) : (b))

int sosfilters(double* inputData, int nInputChannels, int nSamples, double* coeffs,
	       int nFilterChannels, double* state, double* outputData,
	       int nOutputChannels) {
  if (0 == nInputChannels * nSamples)
    return -1;
  if (nOutputChannels != max(nInputChannels, nFilterChannels))
    return -2;
  double *a0, *a1, *a2, *b1, *b2;
  a0 = coeffs;
  a1 = coeffs + nFilterChannels;
  a2 = coeffs + 2 * nFilterChannels;
  b1 = coeffs + 3 * nFilterChannels;
  b2 = coeffs + 4 * nFilterChannels;
  double *state1, *state2;
  state1 = state;
  state2 = state + nOutputChannels;
  int i, n;
  double in, *output;

  if (nInputChannels == 1) {
    for (n = 0; n < nSamples; n++){
      in = inputData[n];
      output = &outputData[n*nOutputChannels];
      for (i=0; i<nOutputChannels; i++){
	output[i] = a0[i] * in                     + state1[i];
	state1[i] = a1[i] * in - b1[i] * output[i] + state2[i];
	state2[i] = a2[i] * in - b2[i] * output[i];
      }
    }
  } else if (nFilterChannels == 1){
    for (n=0; n<nSamples; n++){
      output = &outputData[n*nOutputChannels];
      for (i=0; i<nOutputChannels; i++){
	in = inputData[n*nInputChannels + i];
	output[i] = a0[0] * in                     + state1[i];
	state1[i] = a1[0] * in - b1[0] * output[i] + state2[i];
	state2[i] = a2[0] * in - b2[0] * output[i];
      }
    }
  } else {
    for (n=0; n<nSamples; n++){
      output = &outputData[n*nOutputChannels];
      for (i=0; i<nOutputChannels; i++){
	in = inputData[n*nInputChannels + i];
	output[i] = a0[i] * in                     + state1[i];
	state1[i] = a1[i] * in - b1[i] * output[i] + state2[i];
	state2[i] = a2[i] * in - b2[i] * output[i];
      }
    }
  }

  return 0;
}
