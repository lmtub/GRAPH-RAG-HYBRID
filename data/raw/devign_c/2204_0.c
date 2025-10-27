static void update(Real288_internal *glob)

{

    float buffer1[40], temp1[37];

    float buffer2[8], temp2[11];



    memcpy(buffer1     , glob->output + 20, 20*sizeof(*buffer1));

    memcpy(buffer1 + 20, glob->output     , 20*sizeof(*buffer1));



    do_hybrid_window(36, 40, 35, buffer1, temp1, glob->st1a, glob->st1b,

                     syn_window);



    if (eval_lpc_coeffs(temp1, glob->st1, 36))

        colmult(glob->pr1, glob->st1, table1a, 36);



    memcpy(buffer2    , glob->history + 4, 4*sizeof(*buffer2));

    memcpy(buffer2 + 4, glob->history    , 4*sizeof(*buffer2));



    do_hybrid_window(10, 8, 20, buffer2, temp2, glob->st2a, glob->st2b,

                     gain_window);



    if (eval_lpc_coeffs(temp2, glob->st2, 10))

        colmult(glob->pr2, glob->st2, table2a, 10);

}
