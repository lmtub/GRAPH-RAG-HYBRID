static void pre_calc_cosmod(DCAContext * s)

{

    int i, j, k;

    static int cosmod_inited = 0;



    if(cosmod_inited) return;

    for (j = 0, k = 0; k < 16; k++)

        for (i = 0; i < 16; i++)

            cos_mod[j++] = cos((2 * i + 1) * (2 * k + 1) * M_PI / 64);



    for (k = 0; k < 16; k++)

        for (i = 0; i < 16; i++)

            cos_mod[j++] = cos((i) * (2 * k + 1) * M_PI / 32);



    for (k = 0; k < 16; k++)

        cos_mod[j++] = 0.25 / (2 * cos((2 * k + 1) * M_PI / 128));



    for (k = 0; k < 16; k++)

        cos_mod[j++] = -0.25 / (2.0 * sin((2 * k + 1) * M_PI / 128));



    cosmod_inited = 1;

}
