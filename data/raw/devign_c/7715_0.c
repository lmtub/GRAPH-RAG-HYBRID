static void gain_compensate(COOKContext *q, cook_gains *gains_ptr,

                            float* previous_buffer)

{

    const float fc = q->pow2tab[gains_ptr->previous[0] + 63];

    float *buffer = q->mono_mdct_output;

    int i;



    /* Overlap with the previous block. */

    for(i=0 ; i<q->samples_per_channel ; i++) {

        buffer[i] *= fc;

        buffer[i] += previous_buffer[i];

    }



    /* Apply gain profile */

    for (i = 0; i < 8; i++) {

        if (gains_ptr->now[i] || gains_ptr->now[i + 1])

            interpolate(q, &buffer[q->gain_size_factor * i],

                        gains_ptr->now[i], gains_ptr->now[i + 1]);

    }



    /* Save away the current to be previous block. */

    memcpy(previous_buffer, buffer+q->samples_per_channel,

           sizeof(float)*q->samples_per_channel);

}
