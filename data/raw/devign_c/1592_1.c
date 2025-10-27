static void adaptive_gain_control(float *out, const float *in,

                                  const float *speech_synth,

                                  int size, float alpha, float *gain_mem)

{

    int i;

    float speech_energy = 0.0, postfilter_energy = 0.0, gain_scale_factor;

    float mem = *gain_mem;



    for (i = 0; i < size; i++) {

        speech_energy     += fabsf(speech_synth[i]);

        postfilter_energy += fabsf(in[i]);

    }

    gain_scale_factor = (1.0 - alpha) * speech_energy / postfilter_energy;



    for (i = 0; i < size; i++) {

        mem = alpha * mem + gain_scale_factor;

        out[i] = in[i] * mem;

    }



    *gain_mem = mem;

}
