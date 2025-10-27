int ff_ccitt_unpack(AVCodecContext *avctx,

                    const uint8_t *src, int srcsize,

                    uint8_t *dst, int height, int stride,

                    enum TiffCompr compr, int opts)

{

    int j;

    GetBitContext gb;

    int *runs, *ref, *runend;

    int ret;

    int runsize= avctx->width + 2;



    runs = av_malloc(runsize * sizeof(runs[0]));

    ref  = av_malloc(runsize * sizeof(ref[0]));

    ref[0] = avctx->width;

    ref[1] = 0;

    ref[2] = 0;

    init_get_bits(&gb, src, srcsize*8);

    for(j = 0; j < height; j++){

        runend = runs + runsize;

        if(compr == TIFF_G4){

            ret = decode_group3_2d_line(avctx, &gb, avctx->width, runs, runend, ref);

            if(ret < 0){

                av_free(runs);

                av_free(ref);

                return -1;

            }

        }else{

            int g3d1 = (compr == TIFF_G3) && !(opts & 1);

            if(compr!=TIFF_CCITT_RLE && find_group3_syncmarker(&gb, srcsize*8) < 0)

                break;

            if(compr==TIFF_CCITT_RLE || g3d1 || get_bits1(&gb))

                ret = decode_group3_1d_line(avctx, &gb, avctx->width, runs, runend);

            else

                ret = decode_group3_2d_line(avctx, &gb, avctx->width, runs, runend, ref);

            if(compr==TIFF_CCITT_RLE)

                align_get_bits(&gb);

        }

        if(ret < 0){

            put_line(dst, stride, avctx->width, ref);

        }else{

            put_line(dst, stride, avctx->width, runs);

            FFSWAP(int*, runs, ref);

        }

        dst += stride;

    }

    av_free(runs);

    av_free(ref);

    return 0;

}
