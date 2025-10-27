static void joint_decode(COOKContext *q, float* mlt_buffer1,

                         float* mlt_buffer2) {

    int i,j;

    int decouple_tab[SUBBAND_SIZE];

    float decode_buffer[2048];  //Only 1060 might be needed.

    int idx, cpl_tmp,tmp_idx;

    float f1,f2;

    float* cplscale;



    memset(decouple_tab, 0, sizeof(decouple_tab));

    memset(decode_buffer, 0, sizeof(decode_buffer));



    /* Make sure the buffers are zeroed out. */

    memset(mlt_buffer1,0, 1024*sizeof(float));

    memset(mlt_buffer2,0, 1024*sizeof(float));

    decouple_info(q, decouple_tab);

    mono_decode(q, decode_buffer);



    /* The two channels are stored interleaved in decode_buffer. */

    for (i=0 ; i<q->js_subband_start ; i++) {

        for (j=0 ; j<SUBBAND_SIZE ; j++) {

            mlt_buffer1[i*20+j] = decode_buffer[i*40+j];

            mlt_buffer2[i*20+j] = decode_buffer[i*40+20+j];

        }

    }



    /* When we reach js_subband_start (the higher frequencies)

       the coefficients are stored in a coupling scheme. */

    idx = (1 << q->js_vlc_bits) - 1;

    if (q->js_subband_start < q->subbands) {

        for (i=0 ; i<q->subbands ; i++) {

            cpl_tmp = cplband[i + q->js_subband_start];

            idx -=decouple_tab[cpl_tmp];

            cplscale = (float*)cplscales[q->js_vlc_bits-2];  //choose decoupler table

            f1 = cplscale[decouple_tab[cpl_tmp]];

            f2 = cplscale[idx-1];

            for (j=0 ; j<SUBBAND_SIZE ; j++) {

                tmp_idx = ((2*q->js_subband_start + i)*20)+j;

                mlt_buffer1[20*(i+q->js_subband_start) + j] = f1 * decode_buffer[tmp_idx];

                mlt_buffer2[20*(i+q->js_subband_start) + j] = f2 * decode_buffer[tmp_idx];

            }

            idx = (1 << q->js_vlc_bits) - 1;

        }

    }

}
