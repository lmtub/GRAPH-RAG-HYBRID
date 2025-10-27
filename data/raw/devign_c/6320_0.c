static void yuv2yuvX_c(SwsContext *c, const int16_t *lumFilter,

                       const int16_t **lumSrc, int lumFilterSize,

                       const int16_t *chrFilter, const int16_t **chrUSrc,

                       const int16_t **chrVSrc,

                       int chrFilterSize, const int16_t **alpSrc,

                       uint8_t *dest, uint8_t *uDest, uint8_t *vDest,

                       uint8_t *aDest, int dstW, int chrDstW)

{

    //FIXME Optimize (just quickly written not optimized..)

    int i;

    for (i=0; i<dstW; i++) {

        int val=1<<18;

        int j;

        for (j=0; j<lumFilterSize; j++)

            val += lumSrc[j][i] * lumFilter[j];



        dest[i]= av_clip_uint8(val>>19);

    }



    if (uDest)

        for (i=0; i<chrDstW; i++) {

            int u=1<<18;

            int v=1<<18;

            int j;

            for (j=0; j<chrFilterSize; j++) {

                u += chrUSrc[j][i] * chrFilter[j];

                v += chrVSrc[j][i] * chrFilter[j];

            }



            uDest[i]= av_clip_uint8(u>>19);

            vDest[i]= av_clip_uint8(v>>19);

        }



    if (CONFIG_SWSCALE_ALPHA && aDest)

        for (i=0; i<dstW; i++) {

            int val=1<<18;

            int j;

            for (j=0; j<lumFilterSize; j++)

                val += alpSrc[j][i] * lumFilter[j];



            aDest[i]= av_clip_uint8(val>>19);

        }

}
