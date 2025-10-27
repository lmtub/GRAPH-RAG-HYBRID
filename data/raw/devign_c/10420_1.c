static void msmpeg4_encode_dc(MpegEncContext * s, int level, int n, int *dir_ptr)

{

    int sign, code;

    int pred, extquant;

    int extrabits = 0;



    int16_t *dc_val;

    pred = ff_msmpeg4_pred_dc(s, n, &dc_val, dir_ptr);



    /* update predictor */

    if (n < 4) {

        *dc_val = level * s->y_dc_scale;

    } else {

        *dc_val = level * s->c_dc_scale;

    }



    /* do the prediction */

    level -= pred;



    if(s->msmpeg4_version<=2){

        if (n < 4) {

            put_bits(&s->pb,

                     ff_v2_dc_lum_table[level + 256][1],

                     ff_v2_dc_lum_table[level + 256][0]);

        }else{

            put_bits(&s->pb,

                     ff_v2_dc_chroma_table[level + 256][1],

                     ff_v2_dc_chroma_table[level + 256][0]);

        }

    }else{

        sign = 0;

        if (level < 0) {

            level = -level;

            sign = 1;

        }

        code = level;

        if (code > DC_MAX)

            code = DC_MAX;

        else if( s->msmpeg4_version>=6 ) {

            if( s->qscale == 1 ) {

                extquant = (level + 3) & 0x3;

                code  = ((level+3)>>2);

            } else if( s->qscale == 2 ) {

                extquant = (level + 1) & 0x1;

                code  = ((level+1)>>1);

            }

        }



        if (s->dc_table_index == 0) {

            if (n < 4) {

                put_bits(&s->pb, ff_table0_dc_lum[code][1], ff_table0_dc_lum[code][0]);

            } else {

                put_bits(&s->pb, ff_table0_dc_chroma[code][1], ff_table0_dc_chroma[code][0]);

            }

        } else {

            if (n < 4) {

                put_bits(&s->pb, ff_table1_dc_lum[code][1], ff_table1_dc_lum[code][0]);

            } else {

                put_bits(&s->pb, ff_table1_dc_chroma[code][1], ff_table1_dc_chroma[code][0]);

            }

        }



        if(s->msmpeg4_version>=6 && s->qscale<=2)

            extrabits = 3 - s->qscale;



        if (code == DC_MAX)

            put_bits(&s->pb, 8 + extrabits, level);

        else if(extrabits > 0)//== VC1 && s->qscale<=2

            put_bits(&s->pb, extrabits, extquant);



        if (level != 0) {

            put_bits(&s->pb, 1, sign);

        }

    }

}
