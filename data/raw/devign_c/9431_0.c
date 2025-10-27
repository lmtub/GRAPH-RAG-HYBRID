void ff_atrac_iqmf(float *inlo, float *inhi, unsigned int nIn, float *pOut, float *delayBuf, float *temp)

{

    int   i, j;

    float   *p1, *p3;



    memcpy(temp, delayBuf, 46*sizeof(float));



    p3 = temp + 46;



    /* loop1 */

    for(i=0; i<nIn; i+=2){

        p3[2*i+0] = inlo[i  ] + inhi[i  ];

        p3[2*i+1] = inlo[i  ] - inhi[i  ];

        p3[2*i+2] = inlo[i+1] + inhi[i+1];

        p3[2*i+3] = inlo[i+1] - inhi[i+1];

    }



    /* loop2 */

    p1 = temp;

    for (j = nIn; j != 0; j--) {

        float s1 = 0.0;

        float s2 = 0.0;



        for (i = 0; i < 48; i += 2) {

            s1 += p1[i] * qmf_window[i];

            s2 += p1[i+1] * qmf_window[i+1];

        }



        pOut[0] = s2;

        pOut[1] = s1;



        p1 += 2;

        pOut += 2;

    }



    /* Update the delay buffer. */

    memcpy(delayBuf, temp + nIn*2, 46*sizeof(float));

}
