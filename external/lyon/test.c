#include <stdio.h>
#include <stdlib.h>

#include "lyon.h"

int main()
{
  int nSamples = 5;
  int nChannels = 2;
  int nStages = 3;
  double *signal = (double *)malloc(sizeof(double) * nSamples);
  double *coeffs = (double *)malloc(sizeof(double) * nChannels * 5);
  double *state = (double *)malloc(sizeof(double) * nChannels * 2);
  double *output = (double *)malloc(sizeof(double) * nChannels * nSamples);
  double *agcParams = (double *)malloc(sizeof(double) * 2 * nStages);
  double *agcState = (double *)malloc(sizeof(double) * nChannels * nStages);
  double *agcOut = (double *)malloc(sizeof(double) * nChannels * nSamples);

  for (int i = 0; i < nSamples; ++i)
    signal[i] = -0.5 + i / (double) 10.0 - i * i / (double) 10.0;
  for (int i = 0; i < nChannels * 5; ++i)
    coeffs[i] = i / (double) 100.0;
  for (int i = 0; i < nChannels * 2; ++i)
    state[i] = 0.0;
  for (int i = 0; i < 2 * nStages; ++i)
    agcParams[i] = 0.5;
  for (int i = 0; i < nChannels * nStages; ++i)
    agcState[i] = 0.0;
  printf("Input:\n");
  for (int i = 0; i < nSamples; ++i)
    printf("%2.2f\t", signal[i]);
  printf("\n");

  printf("Testing soscascade...\n");
  int result = soscascade(signal, nSamples, coeffs, state,
			  nChannels, output);
  if (0 != result)
    return result;
  printf("State:\n");
  for (int i = 0; i < nChannels; ++i) {
    for (int j = 0; j < 2; ++j)
      printf("%2.2f\t", state[j * nChannels + i]);
    printf("\n");
  }
  printf("Output:\n");
  for (int i = 0; i < nChannels; ++i) {
    for (int j = 0; j < nSamples; ++j)
      printf("%2.2f\t", output[j * nChannels + i]);
    printf("\n");
  }

  printf("Testing agc...\n");
  result = agc(output, nChannels, nSamples, nStages, agcParams, agcState, agcOut);
  if (0 != result)
    return result;
  printf("State:\n");
  for (int i = 0; i < nChannels; ++i) {
    for (int j = 0; j < nStages; ++j)
      printf("%2.2f\t", agcState[j * nChannels + i]);
    printf("\n");
  }
  printf("Output:\n");
  for (int i = 0; i < nChannels; ++i) {
    for (int j = 0; j < nSamples; ++j)
      printf("%2.2f\t", agcOut[j * nChannels + i]);
    printf("\n");
  }

  printf("Testing sosfilters...\n");
  result = sosfilters(agcOut, nChannels, nSamples, coeffs, nChannels,
		      state, output, nChannels);
  if (0 != result)
    return result;
  printf("State:\n");
  for (int i = 0; i < nChannels; ++i) {
    for (int j = 0; j < 2; ++j)
      printf("%2.2f\t", state[j * nChannels + i]);
    printf("\n");
  }
  printf("Output:\n");
  for (int i = 0; i < nChannels; ++i) {
    for (int j = 0; j < nSamples; ++j)
      printf("%2.2f\t", output[j * nChannels + i]);
    printf("\n");
  }

  free(signal);
  free(coeffs);
  free(state);
  free(output);
  free(agcParams);
  free(agcState);
  free(agcOut);
  return 0;
}
