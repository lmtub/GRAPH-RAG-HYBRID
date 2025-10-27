static void modified_levinson_durbin(int *window, int window_entries,

        int *out, int out_entries, int channels, int *tap_quant)

{

    int i;

    int *state = av_calloc(window_entries, sizeof(*state));



    memcpy(state, window, 4* window_entries);



    for (i = 0; i < out_entries; i++)

    {

        int step = (i+1)*channels, k, j;

        double xx = 0.0, xy = 0.0;

#if 1

        int *x_ptr = &(window[step]);

        int *state_ptr = &(state[0]);

        j = window_entries - step;

        for (;j>0;j--,x_ptr++,state_ptr++)

        {

            double x_value = *x_ptr;

            double state_value = *state_ptr;

            xx += state_value*state_value;

            xy += x_value*state_value;

        }

#else

        for (j = 0; j <= (window_entries - step); j++);

        {

            double stepval = window[step+j];

            double stateval = window[j];

//            xx += (double)window[j]*(double)window[j];

//            xy += (double)window[step+j]*(double)window[j];

            xx += stateval*stateval;

            xy += stepval*stateval;

        }

#endif

        if (xx == 0.0)

            k = 0;

        else

            k = (int)(floor(-xy/xx * (double)LATTICE_FACTOR / (double)(tap_quant[i]) + 0.5));



        if (k > (LATTICE_FACTOR/tap_quant[i]))

            k = LATTICE_FACTOR/tap_quant[i];

        if (-k > (LATTICE_FACTOR/tap_quant[i]))

            k = -(LATTICE_FACTOR/tap_quant[i]);



        out[i] = k;

        k *= tap_quant[i];



#if 1

        x_ptr = &(window[step]);

        state_ptr = &(state[0]);

        j = window_entries - step;

        for (;j>0;j--,x_ptr++,state_ptr++)

        {

            int x_value = *x_ptr;

            int state_value = *state_ptr;

            *x_ptr = x_value + shift_down(k*state_value,LATTICE_SHIFT);

            *state_ptr = state_value + shift_down(k*x_value, LATTICE_SHIFT);

        }

#else

        for (j=0; j <= (window_entries - step); j++)

        {

            int stepval = window[step+j];

            int stateval=state[j];

            window[step+j] += shift_down(k * stateval, LATTICE_SHIFT);

            state[j] += shift_down(k * stepval, LATTICE_SHIFT);

        }

#endif

    }



    av_free(state);

}
