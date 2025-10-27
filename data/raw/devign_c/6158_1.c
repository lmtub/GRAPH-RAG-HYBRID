static int scan_file(AVFormatContext *avctx, AVStream *vst, AVStream *ast, int file)

{

    MlvContext *mlv = avctx->priv_data;

    AVIOContext *pb = mlv->pb[file];

    int ret;

    while (!avio_feof(pb)) {

        int type;

        unsigned int size;

        type = avio_rl32(pb);

        size = avio_rl32(pb);

        avio_skip(pb, 8); //timestamp

        if (size < 16)

            break;

        size -= 16;

        if (vst && type == MKTAG('R','A','W','I') && size >= 164) {

            vst->codec->width  = avio_rl16(pb);

            vst->codec->height = avio_rl16(pb);




            if (avio_rl32(pb) != 1)

                avpriv_request_sample(avctx, "raw api version");

            avio_skip(pb, 20); // pointer, width, height, pitch, frame_size

            vst->codec->bits_per_coded_sample = avio_rl32(pb);









            avio_skip(pb, 8 + 16 + 24); // black_level, white_level, xywh, active_area, exposure_bias

            if (avio_rl32(pb) != 0x2010100) /* RGGB */

                avpriv_request_sample(avctx, "cfa_pattern");

            avio_skip(pb, 80); // calibration_illuminant1, color_matrix1, dynamic_range

            vst->codec->pix_fmt  = AV_PIX_FMT_BAYER_RGGB16LE;

            vst->codec->codec_tag = MKTAG('B', 'I', 'T', 16);

            size -= 164;

        } else if (ast && type == MKTAG('W', 'A', 'V', 'I') && size >= 16) {

            ret = ff_get_wav_header(avctx, pb, ast->codec, 16, 0);



            size -= 16;

        } else if (type == MKTAG('I','N','F','O')) {

            if (size > 0)

                read_string(avctx, pb, "info", size);

            continue;

        } else if (type == MKTAG('I','D','N','T') && size >= 36) {

            read_string(avctx, pb, "cameraName", 32);

            read_uint32(avctx, pb, "cameraModel", "0x%"PRIx32);

            size -= 36;

            if (size >= 32) {

                read_string(avctx, pb, "cameraSerial", 32);

                size -= 32;


        } else if (type == MKTAG('L','E','N','S') && size >= 48) {

            read_uint16(avctx, pb, "focalLength", "%i");

            read_uint16(avctx, pb, "focalDist", "%i");

            read_uint16(avctx, pb, "aperture", "%i");

            read_uint8(avctx, pb, "stabilizerMode", "%i");

            read_uint8(avctx, pb, "autofocusMode", "%i");

            read_uint32(avctx, pb, "flags", "0x%"PRIx32);

            read_uint32(avctx, pb, "lensID", "%"PRIi32);

            read_string(avctx, pb, "lensName", 32);

            size -= 48;

            if (size >= 32) {

                read_string(avctx, pb, "lensSerial", 32);

                size -= 32;


        } else if (vst && type == MKTAG('V', 'I', 'D', 'F') && size >= 4) {

            uint64_t pts = avio_rl32(pb);

            ff_add_index_entry(&vst->index_entries, &vst->nb_index_entries, &vst->index_entries_allocated_size,

                               avio_tell(pb) - 20, pts, file, 0, AVINDEX_KEYFRAME);

            size -= 4;

        } else if (ast && type == MKTAG('A', 'U', 'D', 'F') && size >= 4) {

            uint64_t pts = avio_rl32(pb);

            ff_add_index_entry(&ast->index_entries, &ast->nb_index_entries, &ast->index_entries_allocated_size,

                               avio_tell(pb) - 20, pts, file, 0, AVINDEX_KEYFRAME);

            size -= 4;

        } else if (vst && type == MKTAG('W','B','A','L') && size >= 28) {

            read_uint32(avctx, pb, "wb_mode", "%"PRIi32);

            read_uint32(avctx, pb, "kelvin", "%"PRIi32);

            read_uint32(avctx, pb, "wbgain_r", "%"PRIi32);

            read_uint32(avctx, pb, "wbgain_g", "%"PRIi32);

            read_uint32(avctx, pb, "wbgain_b", "%"PRIi32);

            read_uint32(avctx, pb, "wbs_gm", "%"PRIi32);

            read_uint32(avctx, pb, "wbs_ba", "%"PRIi32);

            size -= 28;

        } else if (type == MKTAG('R','T','C','I') && size >= 20) {

            char str[32];

            struct tm time = { 0 };

            time.tm_sec    = avio_rl16(pb);

            time.tm_min    = avio_rl16(pb);

            time.tm_hour   = avio_rl16(pb);

            time.tm_mday   = avio_rl16(pb);

            time.tm_mon    = avio_rl16(pb);

            time.tm_year   = avio_rl16(pb);

            time.tm_wday   = avio_rl16(pb);

            time.tm_yday   = avio_rl16(pb);

            time.tm_isdst  = avio_rl16(pb);

            avio_skip(pb, 2);

            if (strftime(str, sizeof(str), "%Y-%m-%d %H:%M:%S", &time))

                av_dict_set(&avctx->metadata, "time", str, 0);

            size -= 20;

        } else if (type == MKTAG('E','X','P','O') && size >= 16) {

            av_dict_set(&avctx->metadata, "isoMode", avio_rl32(pb) ? "auto" : "manual", 0);

            read_uint32(avctx, pb, "isoValue", "%"PRIi32);

            read_uint32(avctx, pb, "isoAnalog", "%"PRIi32);

            read_uint32(avctx, pb, "digitalGain", "%"PRIi32);

            size -= 16;

            if (size >= 8) {

                read_uint64(avctx, pb, "shutterValue", "%"PRIi64);

                size -= 8;


        } else if (type == MKTAG('S','T','Y','L') && size >= 36) {

            read_uint32(avctx, pb, "picStyleId", "%"PRIi32);

            read_uint32(avctx, pb, "contrast", "%"PRIi32);

            read_uint32(avctx, pb, "sharpness", "%"PRIi32);

            read_uint32(avctx, pb, "saturation", "%"PRIi32);

            read_uint32(avctx, pb, "colortone", "%"PRIi32);

            read_string(avctx, pb, "picStyleName", 16);

            size -= 36;

        } else if (type == MKTAG('M','A','R','K')) {

        } else if (type == MKTAG('N','U','L','L')) {

        } else if (type == MKTAG('M','L','V','I')) { /* occurs when MLV and Mnn files are concatenated */

        } else {

            av_log(avctx, AV_LOG_INFO, "unsupported tag %c%c%c%c, size %u\n", type&0xFF, (type>>8)&0xFF, (type>>16)&0xFF, (type>>24)&0xFF, size);


        avio_skip(pb, size);


    return 0;
