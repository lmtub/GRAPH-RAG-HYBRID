static int rv30_decode_intra_types(RV34DecContext *r, GetBitContext *gb, int8_t *dst)

{

    int i, j, k;



    for(i = 0; i < 4; i++, dst += r->intra_types_stride - 4){

        for(j = 0; j < 4; j+= 2){

            int code = svq3_get_ue_golomb(gb) << 1;

            if(code >= 81*2){

                av_log(r->s.avctx, AV_LOG_ERROR, "Incorrect intra prediction code\n");

                return -1;

            }

            for(k = 0; k < 2; k++){

                int A = dst[-r->intra_types_stride] + 1;

                int B = dst[-1] + 1;

                *dst++ = rv30_itype_from_context[A * 90 + B * 9 + rv30_itype_code[code + k]];

                if(dst[-1] == 9){

                    av_log(r->s.avctx, AV_LOG_ERROR, "Incorrect intra prediction mode\n");

                    return -1;

                }

            }

        }

    }

    return 0;

}
