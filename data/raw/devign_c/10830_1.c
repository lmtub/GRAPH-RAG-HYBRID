static int dct_quantize_c(MpegEncContext *s, 

                        DCTELEM *block, int n,

                        int qscale)

{

    int i, j, level, last_non_zero, q;

    const int *qmat;

    int minLevel, maxLevel;



    if(s->avctx!=NULL && s->avctx->codec->id==CODEC_ID_MPEG4){

	/* mpeg4 */

        minLevel= -2048;

	maxLevel= 2047;

    }else if(s->out_format==FMT_MPEG1){

	/* mpeg1 */

        minLevel= -255;

	maxLevel= 255;

    }else if(s->out_format==FMT_MJPEG){

	/* (m)jpeg */

        minLevel= -1023;

	maxLevel= 1023;

    }else{

	/* h263 / msmpeg4 */

        minLevel= -128;

	maxLevel= 127;

    }



    av_fdct (block);



    /* we need this permutation so that we correct the IDCT

       permutation. will be moved into DCT code */

    block_permute(block);



    if (s->mb_intra) {

        if (n < 4)

            q = s->y_dc_scale;

        else

            q = s->c_dc_scale;

        q = q << 3;

        

        /* note: block[0] is assumed to be positive */

        block[0] = (block[0] + (q >> 1)) / q;

        i = 1;

        last_non_zero = 0;

        if (s->out_format == FMT_H263) {

            qmat = s->q_non_intra_matrix;

        } else {

            qmat = s->q_intra_matrix;

        }

    } else {

        i = 0;

        last_non_zero = -1;

        qmat = s->q_non_intra_matrix;

    }



    for(;i<64;i++) {

        j = zigzag_direct[i];

        level = block[j];

        level = level * qmat[j];

#ifdef PARANOID

        {

            static int count = 0;

            int level1, level2, qmat1;

            double val;

            if (qmat == s->q_non_intra_matrix) {

                qmat1 = default_non_intra_matrix[j] * s->qscale;

            } else {

                qmat1 = default_intra_matrix[j] * s->qscale;

            }

            if (av_fdct != jpeg_fdct_ifast)

                val = ((double)block[j] * 8.0) / (double)qmat1;

            else

                val = ((double)block[j] * 8.0 * 2048.0) / 

                    ((double)qmat1 * aanscales[j]);

            level1 = (int)val;

            level2 = level / (1 << (QMAT_SHIFT - 3));

            if (level1 != level2) {

                fprintf(stderr, "%d: quant error qlevel=%d wanted=%d level=%d qmat1=%d qmat=%d wantedf=%0.6f\n", 

                        count, level2, level1, block[j], qmat1, qmat[j],

                        val);

                count++;

            }



        }

#endif

        /* XXX: slight error for the low range. Test should be equivalent to

           (level <= -(1 << (QMAT_SHIFT - 3)) || level >= (1 <<

           (QMAT_SHIFT - 3)))

        */

        if (((level << (31 - (QMAT_SHIFT - 3))) >> (31 - (QMAT_SHIFT - 3))) != 

            level) {

            level = level / (1 << (QMAT_SHIFT - 3));

            /* XXX: currently, this code is not optimal. the range should be:

               mpeg1: -255..255

               mpeg2: -2048..2047

               h263:  -128..127

               mpeg4: -2048..2047

            */

            if (level > maxLevel)

                level = maxLevel;

            else if (level < minLevel)

                level = minLevel;



            block[j] = level;

            last_non_zero = i;

        } else {

            block[j] = 0;

        }

    }

    return last_non_zero;

}
