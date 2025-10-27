static void dss_sp_scale_vector(int32_t *vec, int bits, int size)

{

    int i;



    if (bits < 0)

        for (i = 0; i < size; i++)

            vec[i] = vec[i] >> -bits;

    else

        for (i = 0; i < size; i++)

            vec[i] = vec[i] << bits;

}
