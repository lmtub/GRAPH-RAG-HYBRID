static int generate_fake_vps(QSVEncContext *q, AVCodecContext *avctx)

{

    GetByteContext gbc;

    PutByteContext pbc;



    GetBitContext gb;

    H2645NAL sps_nal = { NULL };

    HEVCSPS sps = { 0 };

    HEVCVPS vps = { 0 };

    uint8_t vps_buf[128], vps_rbsp_buf[128];

    uint8_t *new_extradata;

    unsigned int sps_id;

    int ret, i, type, vps_size;



    if (!avctx->extradata_size) {

        av_log(avctx, AV_LOG_ERROR, "No extradata returned from libmfx\n");

        return AVERROR_UNKNOWN;

    }



    /* parse the SPS */

    ret = ff_h2645_extract_rbsp(avctx->extradata + 4, avctx->extradata_size - 4, &sps_nal);

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error unescaping the SPS buffer\n");

        return ret;

    }



    ret = init_get_bits8(&gb, sps_nal.data, sps_nal.size);

    if (ret < 0) {

        av_freep(&sps_nal.rbsp_buffer);

        return ret;

    }



    get_bits(&gb, 1);

    type = get_bits(&gb, 6);

    if (type != NAL_SPS) {

        av_log(avctx, AV_LOG_ERROR, "Unexpected NAL type in the extradata: %d\n",

               type);

        av_freep(&sps_nal.rbsp_buffer);

        return AVERROR_INVALIDDATA;

    }

    get_bits(&gb, 9);



    ret = ff_hevc_parse_sps(&sps, &gb, &sps_id, 0, NULL, avctx);

    av_freep(&sps_nal.rbsp_buffer);

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error parsing the SPS\n");

        return ret;

    }



    /* generate the VPS */

    vps.vps_max_layers     = 1;

    vps.vps_max_sub_layers = sps.max_sub_layers;

    memcpy(&vps.ptl, &sps.ptl, sizeof(vps.ptl));

    vps.vps_sub_layer_ordering_info_present_flag = 1;

    for (i = 0; i < MAX_SUB_LAYERS; i++) {

        vps.vps_max_dec_pic_buffering[i] = sps.temporal_layer[i].max_dec_pic_buffering;

        vps.vps_num_reorder_pics[i]      = sps.temporal_layer[i].num_reorder_pics;

        vps.vps_max_latency_increase[i]  = sps.temporal_layer[i].max_latency_increase;

    }



    vps.vps_num_layer_sets                  = 1;

    vps.vps_timing_info_present_flag        = sps.vui.vui_timing_info_present_flag;

    vps.vps_num_units_in_tick               = sps.vui.vui_num_units_in_tick;

    vps.vps_time_scale                      = sps.vui.vui_time_scale;

    vps.vps_poc_proportional_to_timing_flag = sps.vui.vui_poc_proportional_to_timing_flag;

    vps.vps_num_ticks_poc_diff_one          = sps.vui.vui_num_ticks_poc_diff_one_minus1 + 1;



    /* generate the encoded RBSP form of the VPS */

    ret = ff_hevc_encode_nal_vps(&vps, sps.vps_id, vps_rbsp_buf, sizeof(vps_rbsp_buf));

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error writing the VPS\n");

        return ret;

    }



    /* escape and add the startcode */

    bytestream2_init(&gbc, vps_rbsp_buf, ret);

    bytestream2_init_writer(&pbc, vps_buf, sizeof(vps_buf));



    bytestream2_put_be32(&pbc, 1);              // startcode

    bytestream2_put_byte(&pbc, NAL_VPS << 1);   // NAL

    bytestream2_put_byte(&pbc, 1);              // header



    while (bytestream2_get_bytes_left(&gbc)) {

        uint32_t b = bytestream2_peek_be24(&gbc);

        if (b <= 3) {

            bytestream2_put_be24(&pbc, 3);

            bytestream2_skip(&gbc, 2);

        } else

            bytestream2_put_byte(&pbc, bytestream2_get_byte(&gbc));

    }



    vps_size = bytestream2_tell_p(&pbc);

    new_extradata = av_mallocz(vps_size + avctx->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);

    if (!new_extradata)

        return AVERROR(ENOMEM);

    memcpy(new_extradata, vps_buf, vps_size);

    memcpy(new_extradata + vps_size, avctx->extradata, avctx->extradata_size);



    av_freep(&avctx->extradata);

    avctx->extradata       = new_extradata;

    avctx->extradata_size += vps_size;



    return 0;

}
