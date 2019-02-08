#pragma once

/**
 * Function assumes that output and state are already allocated.
 * Params:
 *   signal is nSamples samples length.
 *   coeffs is nChannels x 5 matrix.
 *   state is treated as nChannels x 2 matrices. Pass zero for initial run.
 *   outputData is nChannels x nSamples matrix.
 * Returns:
 *   0 in case of success.
 */
int soscascade(double* signal, int nSamples, double* coeffs, double* state,
	       int nChannels, double* outputData);
/**
 * Performs AGC.
 * Params:
 *   inputData shape is nChannels x nSamples
 *   agcParams shape is 2 x nStages (targets;epsilons)
 *   state shape is nChannels x nStages
 *   outputData shape is nChannels x nSamples
 */
int agc(double* inputData, int nChannels, int nSamples, int nStages,
	double *agcParams, double* state, double *outputData);
/**
 * Params:
 *   inputData shape is nInputChannels x nSamples
 *   coeffs shape is nFilterChannels x 5
 *   state shape is nOutputChannels x 2
 *   outputData shape is nOutputChannels x nSamples
 */
int sosfilters(double* inputData, int nInputChannels, int nSamples, double* coeffs,
	       int nFilterChannels, double* state, double* outputData,
	       int nOutputChannels);
