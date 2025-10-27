static void process_param(float *bc, EqParameter *param, float fs)

{

    int i;



    for (i = 0; i <= NBANDS; i++) {

        param[i].lower = i == 0 ? 0 : bands[i - 1];

        param[i].upper = i == NBANDS - 1 ? fs : bands[i];

        param[i].gain  = bc[i];

    }

}
