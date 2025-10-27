void ff_mpc_dequantize_and_synth(MPCContext * c, int maxband, void *data, int channels)

{

    int i, j, ch;

    Band *bands = c->bands;

    int off;

    float mul;



    /* dequantize */

    memset(c->sb_samples, 0, sizeof(c->sb_samples));

    off = 0;

    for(i = 0; i <= maxband; i++, off += SAMPLES_PER_BAND){

        for(ch = 0; ch < 2; ch++){

            if(bands[i].res[ch]){

                j = 0;

                mul = mpc_CC[bands[i].res[ch]] * mpc_SCF[bands[i].scf_idx[ch][0] & 0xFF];

                for(; j < 12; j++)

                    c->sb_samples[ch][j][i] = mul * c->Q[ch][j + off];

                mul = mpc_CC[bands[i].res[ch]] * mpc_SCF[bands[i].scf_idx[ch][1] & 0xFF];

                for(; j < 24; j++)

                    c->sb_samples[ch][j][i] = mul * c->Q[ch][j + off];

                mul = mpc_CC[bands[i].res[ch]] * mpc_SCF[bands[i].scf_idx[ch][2] & 0xFF];

                for(; j < 36; j++)

                    c->sb_samples[ch][j][i] = mul * c->Q[ch][j + off];

            }

        }

        if(bands[i].msf){

            int t1, t2;

            for(j = 0; j < SAMPLES_PER_BAND; j++){

                t1 = c->sb_samples[0][j][i];

                t2 = c->sb_samples[1][j][i];

                c->sb_samples[0][j][i] = t1 + t2;

                c->sb_samples[1][j][i] = t1 - t2;

            }

        }

    }



    mpc_synth(c, data, channels);

}
