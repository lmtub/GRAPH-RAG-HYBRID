static int dxva2_get_decoder_configuration(AVCodecContext *s, const GUID *device_guid,

                                           const DXVA2_VideoDesc *desc,

                                           DXVA2_ConfigPictureDecode *config)

{

    InputStream  *ist = s->opaque;

    int loglevel = (ist->hwaccel_id == HWACCEL_AUTO) ? AV_LOG_VERBOSE : AV_LOG_ERROR;

    DXVA2Context *ctx = ist->hwaccel_ctx;

    unsigned cfg_count = 0, best_score = 0;

    DXVA2_ConfigPictureDecode *cfg_list = NULL;

    DXVA2_ConfigPictureDecode best_cfg = {{0}};

    HRESULT hr;

    int i;



    hr = IDirectXVideoDecoderService_GetDecoderConfigurations(ctx->decoder_service, device_guid, desc, NULL, &cfg_count, &cfg_list);

    if (FAILED(hr)) {

        av_log(NULL, loglevel, "Unable to retrieve decoder configurations\n");

        return AVERROR(EINVAL);

    }



    for (i = 0; i < cfg_count; i++) {

        DXVA2_ConfigPictureDecode *cfg = &cfg_list[i];



        unsigned score;

        if (cfg->ConfigBitstreamRaw == 1)

            score = 1;

        else if (s->codec_id == AV_CODEC_ID_H264 && cfg->ConfigBitstreamRaw == 2)

            score = 2;

        else

            continue;

        if (IsEqualGUID(&cfg->guidConfigBitstreamEncryption, &DXVA2_NoEncrypt))

            score += 16;

        if (score > best_score) {

            best_score = score;

            best_cfg   = *cfg;

        }

    }

    CoTaskMemFree(cfg_list);



    if (!best_score) {

        av_log(NULL, loglevel, "No valid decoder configuration available\n");

        return AVERROR(EINVAL);

    }



    *config = best_cfg;

    return 0;

}
