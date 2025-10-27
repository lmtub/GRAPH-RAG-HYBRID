static inline void downmix_3f_2r_to_stereo(float *samples)

{

    int i;



    for (i = 0; i < 256; i++) {

        samples[i] += (samples[i + 256] + samples[i + 768]);

        samples[i + 256] = (samples[i + 512] + samples[i + 1024]);

        samples[i + 512] = samples[i + 768] = samples[i + 1024] = 0;

    }

}
