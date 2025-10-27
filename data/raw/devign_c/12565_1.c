static void dct_unquantize_mpeg2_c(MpegEncContext *s, 

                                   DCTELEM *block, int n, int qscale)

{

    int i, level, nCoeffs;

    const UINT16 *quant_matrix;



    if(s->alternate_scan) nCoeffs= 64;

    else nCoeffs= s->block_last_index[n]+1;

    

    if (s->mb_intra) {

        if (n < 4) 

            block[0] = block[0] * s->y_dc_scale;

        else

            block[0] = block[0] * s->c_dc_scale;

        quant_matrix = s->intra_matrix;

        for(i=1;i<nCoeffs;i++) {

            int j= zigzag_direct[i];

            level = block[j];

            if (level) {

                if (level < 0) {

                    level = -level;

                    level = (int)(level * qscale * quant_matrix[j]) >> 3;

                    level = -level;

                } else {

                    level = (int)(level * qscale * quant_matrix[j]) >> 3;

                }

#ifdef PARANOID

                if (level < -2048 || level > 2047)

                    fprintf(stderr, "unquant error %d %d\n", i, level);

#endif

                block[j] = level;

            }

        }

    } else {

        int sum=-1;

        i = 0;

        quant_matrix = s->non_intra_matrix;

        for(;i<nCoeffs;i++) {

            int j= zigzag_direct[i];

            level = block[j];

            if (level) {

                if (level < 0) {

                    level = -level;

                    level = (((level << 1) + 1) * qscale *

                             ((int) (quant_matrix[j]))) >> 4;

                    level = -level;

                } else {

                    level = (((level << 1) + 1) * qscale *

                             ((int) (quant_matrix[j]))) >> 4;

                }

#ifdef PARANOID

                if (level < -2048 || level > 2047)

                    fprintf(stderr, "unquant error %d %d\n", i, level);

#endif

                block[j] = level;

                sum+=level;

            }

        }

        block[63]^=sum&1;

    }

}
