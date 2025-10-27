static int mp3_header_decompress(AVBitStreamFilterContext *bsfc, AVCodecContext *avctx, const char *args,

                     uint8_t **poutbuf, int *poutbuf_size,

                     const uint8_t *buf, int buf_size, int keyframe){

    uint32_t header;

    int sample_rate= avctx->sample_rate;

    int sample_rate_index=0;

    int lsf, mpeg25, bitrate_index, frame_size;



    header = (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];

    if(ff_mpa_check_header(header) >= 0){

        *poutbuf= (uint8_t *) buf;

        *poutbuf_size= buf_size;



        return 0;

    }



    header= 0xFFE00000 | ((4-3)<<17) | (1<<16); //FIXME simplify



    lsf     = sample_rate < (24000+32000)/2;

    mpeg25  = sample_rate < (12000+16000)/2;

    header |= (!mpeg25)<<20;

    header |= (!lsf   )<<19;

    if(sample_rate<<(lsf+mpeg25) < (44100+32000)/2)

        sample_rate_index |= 2;

    else if(sample_rate<<(lsf+mpeg25) > (44100+48000)/2)

        sample_rate_index |= 1;



    header |= sample_rate_index<<10;

    sample_rate= mpa_freq_tab[sample_rate_index] >> (lsf + mpeg25); //in case sample rate is a little off



    for(bitrate_index=2; bitrate_index<30; bitrate_index++){

        frame_size = mpa_bitrate_tab[lsf][2][bitrate_index>>1];

        frame_size = (frame_size * 144000) / (sample_rate << lsf) + (bitrate_index&1);

        if(frame_size == buf_size + 4)

            break;

    }

    if(bitrate_index == 30){

        av_log(avctx, AV_LOG_ERROR, "couldnt find bitrate_index\n");

        return -1;

    }



    header |= (bitrate_index&1)<<9;

    header |= (bitrate_index>>1)<<12;

    header |= (avctx->channels==1 ? MPA_MONO : MPA_JSTEREO)<<6;



    *poutbuf_size= buf_size + 4;

    *poutbuf= av_malloc(buf_size + 4 + FF_INPUT_BUFFER_PADDING_SIZE);

    memcpy(*poutbuf + 4, buf, buf_size + FF_INPUT_BUFFER_PADDING_SIZE);



    if(avctx->channels==2){

        if(lsf){

            FFSWAP(int, (*poutbuf)[5], (*poutbuf)[6]);

            header |= ((*poutbuf)[5] & 0xC0)>>2;

        }else{

            header |= (*poutbuf)[5] & 0x30;

        }

    }



    (*poutbuf)[0]= header>>24;

    (*poutbuf)[1]= header>>16;

    (*poutbuf)[2]= header>> 8;

    (*poutbuf)[3]= header    ;



    return 1;

}
