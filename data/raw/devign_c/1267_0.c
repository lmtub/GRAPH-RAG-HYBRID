static int decode_block_progressive(MJpegDecodeContext *s, DCTELEM *block, uint8_t *last_nnz,

                                    int ac_index, int16_t *quant_matrix,

                                    int ss, int se, int Al, int *EOBRUN)

{

    int code, i, j, level, val, run;



    if(*EOBRUN){

        (*EOBRUN)--;

        return 0;

    }

    {OPEN_READER(re, &s->gb)

    for(i=ss;;i++) {

        UPDATE_CACHE(re, &s->gb);

        GET_VLC(code, re, &s->gb, s->vlcs[1][ac_index].table, 9, 2)

        /* Progressive JPEG use AC coeffs from zero and this decoder sets offset 16 by default */

        code -= 16;

        run = ((unsigned) code) >> 4;

        code &= 0xF;

        if(code) {

            i += run;

            if(code > MIN_CACHE_BITS - 16){

                UPDATE_CACHE(re, &s->gb)

            }

            {

                int cache=GET_CACHE(re,&s->gb);

                int sign=(~cache)>>31;

                level = (NEG_USR32(sign ^ cache,code) ^ sign) - sign;

            }



            LAST_SKIP_BITS(re, &s->gb, code)



            if (i >= se) {

                if(i == se){

                    j = s->scantable.permutated[se];

                    block[j] = level * quant_matrix[j] << Al;

                    break;

                }

                av_log(s->avctx, AV_LOG_ERROR, "error count: %d\n", i);

                return -1;

            }

            j = s->scantable.permutated[i];

            block[j] = level * quant_matrix[j] << Al;

        }else{

            if(run == 0xF){// ZRL - skip 15 coefficients

                i += 15;

            }else{

                val = (1 << run);

                if(run){

                    UPDATE_CACHE(re, &s->gb);

                    val += NEG_USR32(GET_CACHE(re, &s->gb), run);

                    LAST_SKIP_BITS(re, &s->gb, run);

                }

                *EOBRUN = val - 1;

                break;

            }

        }

    }

    CLOSE_READER(re, &s->gb)}

    if(i > *last_nnz)

        *last_nnz = i;

    return 0;

}
