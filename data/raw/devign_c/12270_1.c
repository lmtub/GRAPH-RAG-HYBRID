int ff_dca_lbr_parse(DCALbrDecoder *s, uint8_t *data, DCAExssAsset *asset)

{

    struct {

        LBRChunk    lfe;

        LBRChunk    tonal;

        LBRChunk    tonal_grp[5];

        LBRChunk    grid1[DCA_LBR_CHANNELS / 2];

        LBRChunk    hr_grid[DCA_LBR_CHANNELS / 2];

        LBRChunk    ts1[DCA_LBR_CHANNELS / 2];

        LBRChunk    ts2[DCA_LBR_CHANNELS / 2];

    } chunk = { 0 };



    GetByteContext gb;



    int i, ch, sb, sf, ret, group, chunk_id, chunk_len;



    bytestream2_init(&gb, data + asset->lbr_offset, asset->lbr_size);



    // LBR sync word

    if (bytestream2_get_be32(&gb) != DCA_SYNCWORD_LBR) {

        av_log(s->avctx, AV_LOG_ERROR, "Invalid LBR sync word\n");

        return AVERROR_INVALIDDATA;

    }



    // LBR header type

    switch (bytestream2_get_byte(&gb)) {

    case LBR_HEADER_SYNC_ONLY:

        if (!s->sample_rate) {

            av_log(s->avctx, AV_LOG_ERROR, "LBR decoder not initialized\n");

            return AVERROR_INVALIDDATA;

        }

        break;

    case LBR_HEADER_DECODER_INIT:

        if ((ret = parse_decoder_init(s, &gb)) < 0) {

            s->sample_rate = 0;

            return ret;

        }

        break;

    default:

        av_log(s->avctx, AV_LOG_ERROR, "Invalid LBR header type\n");

        return AVERROR_INVALIDDATA;

    }



    // LBR frame chunk header

    chunk_id = bytestream2_get_byte(&gb);

    chunk_len = (chunk_id & 0x80) ? bytestream2_get_be16(&gb) : bytestream2_get_byte(&gb);



    if (chunk_len > bytestream2_get_bytes_left(&gb)) {

        chunk_len = bytestream2_get_bytes_left(&gb);

        av_log(s->avctx, AV_LOG_WARNING, "LBR frame chunk was truncated\n");

        if (s->avctx->err_recognition & AV_EF_EXPLODE)

            return AVERROR_INVALIDDATA;

    }



    bytestream2_init(&gb, gb.buffer, chunk_len);



    switch (chunk_id & 0x7f) {

    case LBR_CHUNK_FRAME:

        if (s->avctx->err_recognition & (AV_EF_CRCCHECK | AV_EF_CAREFUL)) {

            int checksum = bytestream2_get_be16(&gb);

            uint16_t res = chunk_id;

            res += (chunk_len >> 8) & 0xff;

            res += chunk_len & 0xff;

            for (i = 0; i < chunk_len - 2; i++)

                res += gb.buffer[i];

            if (checksum != res) {

                av_log(s->avctx, AV_LOG_WARNING, "Invalid LBR checksum\n");

                if (s->avctx->err_recognition & AV_EF_EXPLODE)

                    return AVERROR_INVALIDDATA;

            }

        } else {

            bytestream2_skip(&gb, 2);

        }

        break;

    case LBR_CHUNK_FRAME_NO_CSUM:

        break;

    default:

        av_log(s->avctx, AV_LOG_ERROR, "Invalid LBR frame chunk ID\n");

        return AVERROR_INVALIDDATA;

    }



    // Clear current frame

    memset(s->quant_levels, 0, sizeof(s->quant_levels));

    memset(s->sb_indices, 0xff, sizeof(s->sb_indices));

    memset(s->sec_ch_sbms, 0, sizeof(s->sec_ch_sbms));

    memset(s->sec_ch_lrms, 0, sizeof(s->sec_ch_lrms));

    memset(s->ch_pres, 0, sizeof(s->ch_pres));

    memset(s->grid_1_scf, 0, sizeof(s->grid_1_scf));

    memset(s->grid_2_scf, 0, sizeof(s->grid_2_scf));

    memset(s->grid_3_avg, 0, sizeof(s->grid_3_avg));

    memset(s->grid_3_scf, 0, sizeof(s->grid_3_scf));

    memset(s->grid_3_pres, 0, sizeof(s->grid_3_pres));

    memset(s->tonal_scf, 0, sizeof(s->tonal_scf));

    memset(s->lfe_data, 0, sizeof(s->lfe_data));

    s->part_stereo_pres = 0;

    s->framenum = (s->framenum + 1) & 31;



    for (ch = 0; ch < s->nchannels; ch++) {

        for (sb = 0; sb < s->nsubbands / 4; sb++) {

            s->part_stereo[ch][sb][0] = s->part_stereo[ch][sb][4];

            s->part_stereo[ch][sb][4] = 16;

        }

    }



    memset(s->lpc_coeff[s->framenum & 1], 0, sizeof(s->lpc_coeff[0]));



    for (group = 0; group < 5; group++) {

        for (sf = 0; sf < 1 << group; sf++) {

            int sf_idx = ((s->framenum << group) + sf) & 31;

            s->tonal_bounds[group][sf_idx][0] =

            s->tonal_bounds[group][sf_idx][1] = s->ntones;

        }

    }



    // Parse chunk headers

    while (bytestream2_get_bytes_left(&gb) > 0) {

        chunk_id = bytestream2_get_byte(&gb);

        chunk_len = (chunk_id & 0x80) ? bytestream2_get_be16(&gb) : bytestream2_get_byte(&gb);

        chunk_id &= 0x7f;



        if (chunk_len > bytestream2_get_bytes_left(&gb)) {

            chunk_len = bytestream2_get_bytes_left(&gb);

            av_log(s->avctx, AV_LOG_WARNING, "LBR chunk %#x was truncated\n", chunk_id);

            if (s->avctx->err_recognition & AV_EF_EXPLODE)

                return AVERROR_INVALIDDATA;

        }



        switch (chunk_id) {

        case LBR_CHUNK_LFE:

            chunk.lfe.len  = chunk_len;

            chunk.lfe.data = gb.buffer;

            break;



        case LBR_CHUNK_SCF:

        case LBR_CHUNK_TONAL:

        case LBR_CHUNK_TONAL_SCF:

            chunk.tonal.id   = chunk_id;

            chunk.tonal.len  = chunk_len;

            chunk.tonal.data = gb.buffer;

            break;



        case LBR_CHUNK_TONAL_GRP_1:

        case LBR_CHUNK_TONAL_GRP_2:

        case LBR_CHUNK_TONAL_GRP_3:

        case LBR_CHUNK_TONAL_GRP_4:

        case LBR_CHUNK_TONAL_GRP_5:

            i = LBR_CHUNK_TONAL_GRP_5 - chunk_id;

            chunk.tonal_grp[i].id   = i;

            chunk.tonal_grp[i].len  = chunk_len;

            chunk.tonal_grp[i].data = gb.buffer;

            break;



        case LBR_CHUNK_TONAL_SCF_GRP_1:

        case LBR_CHUNK_TONAL_SCF_GRP_2:

        case LBR_CHUNK_TONAL_SCF_GRP_3:

        case LBR_CHUNK_TONAL_SCF_GRP_4:

        case LBR_CHUNK_TONAL_SCF_GRP_5:

            i = LBR_CHUNK_TONAL_SCF_GRP_5 - chunk_id;

            chunk.tonal_grp[i].id   = i;

            chunk.tonal_grp[i].len  = chunk_len;

            chunk.tonal_grp[i].data = gb.buffer;

            break;



        case LBR_CHUNK_RES_GRID_LR:

        case LBR_CHUNK_RES_GRID_LR + 1:

        case LBR_CHUNK_RES_GRID_LR + 2:

            i = chunk_id - LBR_CHUNK_RES_GRID_LR;

            chunk.grid1[i].len  = chunk_len;

            chunk.grid1[i].data = gb.buffer;

            break;



        case LBR_CHUNK_RES_GRID_HR:

        case LBR_CHUNK_RES_GRID_HR + 1:

        case LBR_CHUNK_RES_GRID_HR + 2:

            i = chunk_id - LBR_CHUNK_RES_GRID_HR;

            chunk.hr_grid[i].len  = chunk_len;

            chunk.hr_grid[i].data = gb.buffer;

            break;



        case LBR_CHUNK_RES_TS_1:

        case LBR_CHUNK_RES_TS_1 + 1:

        case LBR_CHUNK_RES_TS_1 + 2:

            i = chunk_id - LBR_CHUNK_RES_TS_1;

            chunk.ts1[i].len  = chunk_len;

            chunk.ts1[i].data = gb.buffer;

            break;



        case LBR_CHUNK_RES_TS_2:

        case LBR_CHUNK_RES_TS_2 + 1:

        case LBR_CHUNK_RES_TS_2 + 2:

            i = chunk_id - LBR_CHUNK_RES_TS_2;

            chunk.ts2[i].len  = chunk_len;

            chunk.ts2[i].data = gb.buffer;

            break;

        }



        bytestream2_skip(&gb, chunk_len);

    }



    // Parse the chunks

    ret = parse_lfe_chunk(s, &chunk.lfe);



    ret |= parse_tonal_chunk(s, &chunk.tonal);



    for (i = 0; i < 5; i++)

        ret |= parse_tonal_group(s, &chunk.tonal_grp[i]);



    for (i = 0; i < (s->nchannels + 1) / 2; i++) {

        int ch1 = i * 2;

        int ch2 = FFMIN(ch1 + 1, s->nchannels - 1);



        if (parse_grid_1_chunk (s, &chunk.grid1  [i], ch1, ch2) < 0 ||

            parse_high_res_grid(s, &chunk.hr_grid[i], ch1, ch2) < 0) {

            ret = -1;

            continue;

        }



        // TS chunks depend on both grids. TS_2 depends on TS_1.

        if (!chunk.grid1[i].len || !chunk.hr_grid[i].len || !chunk.ts1[i].len)

            continue;



        if (parse_ts1_chunk(s, &chunk.ts1[i], ch1, ch2) < 0 ||

            parse_ts2_chunk(s, &chunk.ts2[i], ch1, ch2) < 0) {

            ret = -1;

            continue;

        }

    }



    if (ret < 0 && (s->avctx->err_recognition & AV_EF_EXPLODE))

        return AVERROR_INVALIDDATA;



    return 0;

}
