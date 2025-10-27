static void do_rematrixing(AC3DecodeContext *s)

{

    int bnd, i;

    int end, bndend;

    int tmp0, tmp1;



    end = FFMIN(s->end_freq[1], s->end_freq[2]);



    for(bnd=0; bnd<s->num_rematrixing_bands; bnd++) {

        if(s->rematrixing_flags[bnd]) {

            bndend = FFMIN(end, ff_ac3_rematrix_band_tab[bnd+1]);

            for(i=ff_ac3_rematrix_band_tab[bnd]; i<bndend; i++) {

                tmp0 = s->fixed_coeffs[1][i];

                tmp1 = s->fixed_coeffs[2][i];

                s->fixed_coeffs[1][i] = tmp0 + tmp1;

                s->fixed_coeffs[2][i] = tmp0 - tmp1;

            }

        }

    }

}
