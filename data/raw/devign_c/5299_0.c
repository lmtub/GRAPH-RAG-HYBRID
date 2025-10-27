static int joint_decode(COOKContext *q, COOKSubpacket *p, float *mlt_buffer1,

                         float *mlt_buffer2)

{

    int i, j, ret;

    int decouple_tab[SUBBAND_SIZE];

    float *decode_buffer = q->decode_buffer_0;

    int idx, cpl_tmp;

    float f1, f2;

    const float *cplscale;



    memset(decouple_tab, 0, sizeof(decouple_tab));

    memset(decode_buffer, 0, sizeof(q->decode_buffer_0));



    /* Make sure the buffers are zeroed out. */

    memset(mlt_buffer1, 0, 1024 * sizeof(*mlt_buffer1));

    memset(mlt_buffer2, 0, 1024 * sizeof(*mlt_buffer2));

    decouple_info(q, p, decouple_tab);

    if ((ret = mono_decode(q, p, decode_buffer)) < 0)

        return ret;

    /* The two channels are stored interleaved in decode_buffer. */

    for (i = 0; i < p->js_subband_start; i++) {

        for (j = 0; j < SUBBAND_SIZE; j++) {

            mlt_buffer1[i * 20 + j] = decode_buffer[i * 40 + j];

            mlt_buffer2[i * 20 + j] = decode_buffer[i * 40 + 20 + j];

        }

    }



    /* When we reach js_subband_start (the higher frequencies)

       the coefficients are stored in a coupling scheme. */

    idx = (1 << p->js_vlc_bits) - 1;

    for (i = p->js_subband_start; i < p->subbands; i++) {

        cpl_tmp = cplband[i];

        idx -= decouple_tab[cpl_tmp];

        cplscale = q->cplscales[p->js_vlc_bits - 2];  // choose decoupler table

        f1 = cplscale[decouple_tab[cpl_tmp]];

        f2 = cplscale[idx - 1];

        q->decouple(q, p, i, f1, f2, decode_buffer, mlt_buffer1, mlt_buffer2);

        idx = (1 << p->js_vlc_bits) - 1;

    }

    return 0;

}
