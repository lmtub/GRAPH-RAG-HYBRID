static inline void downmix_dualmono_to_stereo(float *samples)

{

    int i;

    float tmp;



    for (i = 0; i < 256; i++) {

        tmp = samples[i] + samples[i + 256];

        samples[i] = samples[i + 256] = tmp;

    }

}
