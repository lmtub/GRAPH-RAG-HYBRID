static float ppp_pvq_search_c(float *X, int *y, int K, int N)

{

    int i, y_norm = 0;

    float res = 0.0f, xy_norm = 0.0f;



    for (i = 0; i < N; i++)

        res += FFABS(X[i]);



    res = K/(res + FLT_EPSILON);



    for (i = 0; i < N; i++) {

        y[i] = lrintf(res*X[i]);

        y_norm  += y[i]*y[i];

        xy_norm += y[i]*X[i];

        K -= FFABS(y[i]);

    }



    while (K) {

        int max_idx = 0, max_den = 1, phase = FFSIGN(K);

        float max_num = 0.0f;

        y_norm += 1.0f;



        for (i = 0; i < N; i++) {

            /* If the sum has been overshot and the best place has 0 pulses allocated

             * to it, attempting to decrease it further will actually increase the

             * sum. Prevent this by disregarding any 0 positions when decrementing. */

            const int ca = 1 ^ ((y[i] == 0) & (phase < 0));

            const int y_new = y_norm  + 2*phase*FFABS(y[i]);

            float xy_new = xy_norm + 1*phase*FFABS(X[i]);

            xy_new = xy_new * xy_new;

            if (ca && (max_den*xy_new) > (y_new*max_num)) {

                max_den = y_new;

                max_num = xy_new;

                max_idx = i;

            }

        }



        K -= phase;



        phase *= FFSIGN(X[max_idx]);

        xy_norm += 1*phase*X[max_idx];

        y_norm  += 2*phase*y[max_idx];

        y[max_idx] += phase;

    }



    return (float)y_norm;

}
