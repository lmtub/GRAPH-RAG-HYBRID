static int mp3_header_compress(AVBitStreamFilterContext *bsfc, AVCodecContext *avctx, const char *args,

                     uint8_t **poutbuf, int *poutbuf_size,

                     const uint8_t *buf, int buf_size, int keyframe){

    uint32_t header;

    int mode_extension;



    if(avctx->strict_std_compliance > FF_COMPLIANCE_EXPERIMENTAL){

        av_log(avctx, AV_LOG_ERROR, "not standards compliant\n");

        return -1;

    }



    header = (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];

    mode_extension= (header>>4)&3;



    if(ff_mpa_check_header(header) < 0 || (header&0x70000) != 0x30000){

        *poutbuf= (uint8_t *) buf;

        *poutbuf_size= buf_size;



        av_log(avctx, AV_LOG_INFO, "cannot compress %08X\n", header);

        return 0;

    }



    *poutbuf_size= buf_size - 4;

    *poutbuf= av_malloc(buf_size - 4 + FF_INPUT_BUFFER_PADDING_SIZE);

    memcpy(*poutbuf, buf + 4, buf_size - 4 + FF_INPUT_BUFFER_PADDING_SIZE);



    if(avctx->channels==2){

        if((header & (3<<19)) != 3<<19){

            (*poutbuf)[1] &= 0x3F;

            (*poutbuf)[1] |= mode_extension<<6;

            FFSWAP(int, (*poutbuf)[1], (*poutbuf)[2]);

        }else{

            (*poutbuf)[1] &= 0x8F;

            (*poutbuf)[1] |= mode_extension<<4;

        }

    }



    return 1;

}
