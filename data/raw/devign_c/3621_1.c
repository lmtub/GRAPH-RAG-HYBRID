static inline void yuv2nv12XinC(int16_t *lumFilter, int16_t **lumSrc, int lumFilterSize,

                                int16_t *chrFilter, int16_t **chrSrc, int chrFilterSize,

                                uint8_t *dest, uint8_t *uDest, int dstW, int chrDstW, int dstFormat)

{

    //FIXME Optimize (just quickly writen not opti..)

    int i;

    for (i=0; i<dstW; i++)

    {

        int val=1<<18;

        int j;

        for (j=0; j<lumFilterSize; j++)

            val += lumSrc[j][i] * lumFilter[j];



        dest[i]= av_clip_uint8(val>>19);

    }



    if (!uDest)

        return;



    if (dstFormat == PIX_FMT_NV12)

        for (i=0; i<chrDstW; i++)

        {

            int u=1<<18;

            int v=1<<18;

            int j;

            for (j=0; j<chrFilterSize; j++)

            {

                u += chrSrc[j][i] * chrFilter[j];

                v += chrSrc[j][i + 2048] * chrFilter[j];

            }



            uDest[2*i]= av_clip_uint8(u>>19);

            uDest[2*i+1]= av_clip_uint8(v>>19);

        }

    else

        for (i=0; i<chrDstW; i++)

        {

            int u=1<<18;

            int v=1<<18;

            int j;

            for (j=0; j<chrFilterSize; j++)

            {

                u += chrSrc[j][i] * chrFilter[j];

                v += chrSrc[j][i + 2048] * chrFilter[j];

            }



            uDest[2*i]= av_clip_uint8(v>>19);

            uDest[2*i+1]= av_clip_uint8(u>>19);

        }

}
