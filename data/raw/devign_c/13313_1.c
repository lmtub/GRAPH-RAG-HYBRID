static float pvq_band_cost(CeltPVQ *pvq, CeltFrame *f, OpusRangeCoder *rc, int band,

                           float *bits, float lambda)

{

    int i, b = 0;

    uint32_t cm[2] = { (1 << f->blocks) - 1, (1 << f->blocks) - 1 };

    const int band_size = ff_celt_freq_range[band] << f->size;

    float buf[176 * 2], lowband_scratch[176], norm1[176], norm2[176];

    float dist, cost, err_x = 0.0f, err_y = 0.0f;

    float *X = buf;

    float *X_orig = f->block[0].coeffs + (ff_celt_freq_bands[band] << f->size);

    float *Y = (f->channels == 2) ? &buf[176] : NULL;

    float *Y_orig = f->block[1].coeffs + (ff_celt_freq_bands[band] << f->size);

    OPUS_RC_CHECKPOINT_SPAWN(rc);



    memcpy(X, X_orig, band_size*sizeof(float));

    if (Y)

        memcpy(Y, Y_orig, band_size*sizeof(float));



    f->remaining2 = ((f->framebits << 3) - f->anticollapse_needed) - opus_rc_tell_frac(rc) - 1;

    if (band <= f->coded_bands - 1) {

        int curr_balance = f->remaining / FFMIN(3, f->coded_bands - band);

        b = av_clip_uintp2(FFMIN(f->remaining2 + 1, f->pulses[band] + curr_balance), 14);

    }



    if (f->dual_stereo) {

        pvq->encode_band(pvq, f, rc, band, X, NULL, band_size, b / 2, f->blocks, NULL,

                         f->size, norm1, 0, 1.0f, lowband_scratch, cm[0]);



        pvq->encode_band(pvq, f, rc, band, Y, NULL, band_size, b / 2, f->blocks, NULL,

                         f->size, norm2, 0, 1.0f, lowband_scratch, cm[1]);

    } else {

        pvq->encode_band(pvq, f, rc, band, X, Y, band_size, b, f->blocks, NULL, f->size,

                         norm1, 0, 1.0f, lowband_scratch, cm[0] | cm[1]);

    }



    for (i = 0; i < band_size; i++) {

        err_x += (X[i] - X_orig[i])*(X[i] - X_orig[i]);

        err_y += (Y[i] - Y_orig[i])*(Y[i] - Y_orig[i]);

    }



    dist = sqrtf(err_x) + sqrtf(err_y);

    cost = OPUS_RC_CHECKPOINT_BITS(rc)/8.0f;

    *bits += cost;



    OPUS_RC_CHECKPOINT_ROLLBACK(rc);



    return lambda*dist*cost;

}
