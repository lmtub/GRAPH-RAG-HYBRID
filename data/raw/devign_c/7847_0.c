static int imc_decode_frame(AVCodecContext * avctx,

                            void *data, int *data_size,

                            AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;



    IMCContext *q = avctx->priv_data;



    int stream_format_code;

    int imc_hdr, i, j;

    int flag;

    int bits, summer;

    int counter, bitscount;

    LOCAL_ALIGNED_16(uint16_t, buf16, [IMC_BLOCK_SIZE / 2]);



    if (buf_size < IMC_BLOCK_SIZE) {

        av_log(avctx, AV_LOG_ERROR, "imc frame too small!\n");

        return -1;

    }



    q->dsp.bswap16_buf(buf16, (const uint16_t*)buf, IMC_BLOCK_SIZE / 2);



    q->out_samples = data;

    init_get_bits(&q->gb, (const uint8_t*)buf16, IMC_BLOCK_SIZE * 8);



    /* Check the frame header */

    imc_hdr = get_bits(&q->gb, 9);

    if (imc_hdr != IMC_FRAME_ID) {

        av_log(avctx, AV_LOG_ERROR, "imc frame header check failed!\n");

        av_log(avctx, AV_LOG_ERROR, "got %x instead of 0x21.\n", imc_hdr);

        return -1;

    }

    stream_format_code = get_bits(&q->gb, 3);



    if(stream_format_code & 1){

        av_log(avctx, AV_LOG_ERROR, "Stream code format %X is not supported\n", stream_format_code);

        return -1;

    }



//    av_log(avctx, AV_LOG_DEBUG, "stream_format_code = %d\n", stream_format_code);



    if (stream_format_code & 0x04)

        q->decoder_reset = 1;



    if(q->decoder_reset) {

        memset(q->out_samples, 0, sizeof(q->out_samples));

        for(i = 0; i < BANDS; i++)q->old_floor[i] = 1.0;

        for(i = 0; i < COEFFS; i++)q->CWdecoded[i] = 0;

        q->decoder_reset = 0;

    }



    flag = get_bits1(&q->gb);

    imc_read_level_coeffs(q, stream_format_code, q->levlCoeffBuf);



    if (stream_format_code & 0x4)

        imc_decode_level_coefficients(q, q->levlCoeffBuf, q->flcoeffs1, q->flcoeffs2);

    else

        imc_decode_level_coefficients2(q, q->levlCoeffBuf, q->old_floor, q->flcoeffs1, q->flcoeffs2);



    memcpy(q->old_floor, q->flcoeffs1, 32 * sizeof(float));



    counter = 0;

    for (i=0 ; i<BANDS ; i++) {

        if (q->levlCoeffBuf[i] == 16) {

            q->bandWidthT[i] = 0;

            counter++;

        } else

            q->bandWidthT[i] = band_tab[i+1] - band_tab[i];

    }

    memset(q->bandFlagsBuf, 0, BANDS * sizeof(int));

    for(i = 0; i < BANDS-1; i++) {

        if (q->bandWidthT[i])

            q->bandFlagsBuf[i] = get_bits1(&q->gb);

    }



    imc_calculate_coeffs(q, q->flcoeffs1, q->flcoeffs2, q->bandWidthT, q->flcoeffs3, q->flcoeffs5);



    bitscount = 0;

    /* first 4 bands will be assigned 5 bits per coefficient */

    if (stream_format_code & 0x2) {

        bitscount += 15;



        q->bitsBandT[0] = 5;

        q->CWlengthT[0] = 5;

        q->CWlengthT[1] = 5;

        q->CWlengthT[2] = 5;

        for(i = 1; i < 4; i++){

            bits = (q->levlCoeffBuf[i] == 16) ? 0 : 5;

            q->bitsBandT[i] = bits;

            for(j = band_tab[i]; j < band_tab[i+1]; j++) {

                q->CWlengthT[j] = bits;

                bitscount += bits;

            }

        }

    }



    if(bit_allocation (q, stream_format_code, 512 - bitscount - get_bits_count(&q->gb), flag) < 0) {

        av_log(avctx, AV_LOG_ERROR, "Bit allocations failed\n");

        q->decoder_reset = 1;

        return -1;

    }



    for(i = 0; i < BANDS; i++) {

        q->sumLenArr[i] = 0;

        q->skipFlagRaw[i] = 0;

        for(j = band_tab[i]; j < band_tab[i+1]; j++)

            q->sumLenArr[i] += q->CWlengthT[j];

        if (q->bandFlagsBuf[i])

            if( (((band_tab[i+1] - band_tab[i]) * 1.5) > q->sumLenArr[i]) && (q->sumLenArr[i] > 0))

                q->skipFlagRaw[i] = 1;

    }



    imc_get_skip_coeff(q);



    for(i = 0; i < BANDS; i++) {

        q->flcoeffs6[i] = q->flcoeffs1[i];

        /* band has flag set and at least one coded coefficient */

        if (q->bandFlagsBuf[i] && (band_tab[i+1] - band_tab[i]) != q->skipFlagCount[i]){

                q->flcoeffs6[i] *= q->sqrt_tab[band_tab[i+1] - band_tab[i]] /

                                   q->sqrt_tab[(band_tab[i+1] - band_tab[i] - q->skipFlagCount[i])];

        }

    }



    /* calculate bits left, bits needed and adjust bit allocation */

    bits = summer = 0;



    for(i = 0; i < BANDS; i++) {

        if (q->bandFlagsBuf[i]) {

            for(j = band_tab[i]; j < band_tab[i+1]; j++) {

                if(q->skipFlags[j]) {

                    summer += q->CWlengthT[j];

                    q->CWlengthT[j] = 0;

                }

            }

            bits += q->skipFlagBits[i];

            summer -= q->skipFlagBits[i];

        }

    }

    imc_adjust_bit_allocation(q, summer);



    for(i = 0; i < BANDS; i++) {

        q->sumLenArr[i] = 0;



        for(j = band_tab[i]; j < band_tab[i+1]; j++)

            if (!q->skipFlags[j])

                q->sumLenArr[i] += q->CWlengthT[j];

    }



    memset(q->codewords, 0, sizeof(q->codewords));



    if(imc_get_coeffs(q) < 0) {

        av_log(avctx, AV_LOG_ERROR, "Read coefficients failed\n");

        q->decoder_reset = 1;

        return 0;

    }



    if(inverse_quant_coeff(q, stream_format_code) < 0) {

        av_log(avctx, AV_LOG_ERROR, "Inverse quantization of coefficients failed\n");

        q->decoder_reset = 1;

        return 0;

    }



    memset(q->skipFlags, 0, sizeof(q->skipFlags));



    imc_imdct256(q);



    *data_size = COEFFS * sizeof(float);



    return IMC_BLOCK_SIZE;

}
