static inline void float_to_int (float * _f, int16_t * s16, int samples)

{

    int32_t * f = (int32_t *) _f;       // XXX assumes IEEE float format

    int i;



    for (i = 0; i < samples; i++) {

        s16[i] = blah (f[i]);

    }

}
