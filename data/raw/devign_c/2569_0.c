void ff_flac_compute_autocorr(const int32_t *data, int len, int lag,

                              double *autoc)

{

    int i, j;

    double tmp[len + lag + 1];

    double *data1= tmp + lag;



    apply_welch_window(data, len, data1);



    for(j=0; j<lag; j++)

        data1[j-lag]= 0.0;

    data1[len] = 0.0;



    for(j=0; j<lag; j+=2){

        double sum0 = 1.0, sum1 = 1.0;

        for(i=0; i<len; i++){

            sum0 += data1[i] * data1[i-j];

            sum1 += data1[i] * data1[i-j-1];

        }

        autoc[j  ] = sum0;

        autoc[j+1] = sum1;

    }



    if(j==lag){

        double sum = 1.0;

        for(i=0; i<len; i+=2){

            sum += data1[i  ] * data1[i-j  ]

                 + data1[i+1] * data1[i-j+1];

        }

        autoc[j] = sum;

    }

}
