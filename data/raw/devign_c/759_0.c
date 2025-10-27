static int spdif_get_offset_and_codec(AVFormatContext *s,

                                      enum IEC61937DataType data_type,

                                      const char *buf, int *offset,

                                      enum AVCodecID *codec)

{

    AACADTSHeaderInfo aac_hdr;

    GetBitContext gbc;



    switch (data_type & 0xff) {

    case IEC61937_AC3:

        *offset = AC3_FRAME_SIZE << 2;

        *codec = AV_CODEC_ID_AC3;

        break;

    case IEC61937_MPEG1_LAYER1:

        *offset = spdif_mpeg_pkt_offset[1][0];

        *codec = AV_CODEC_ID_MP1;

        break;

    case IEC61937_MPEG1_LAYER23:

        *offset = spdif_mpeg_pkt_offset[1][0];

        *codec = AV_CODEC_ID_MP3;

        break;

    case IEC61937_MPEG2_EXT:

        *offset = 4608;

        *codec = AV_CODEC_ID_MP3;

        break;

    case IEC61937_MPEG2_AAC:

        init_get_bits(&gbc, buf, AAC_ADTS_HEADER_SIZE * 8);

        if (avpriv_aac_parse_header(&gbc, &aac_hdr)) {

            if (s) /* be silent during a probe */

                av_log(s, AV_LOG_ERROR, "Invalid AAC packet in IEC 61937\n");

            return AVERROR_INVALIDDATA;

        }

        *offset = aac_hdr.samples << 2;

        *codec = AV_CODEC_ID_AAC;

        break;

    case IEC61937_MPEG2_LAYER1_LSF:

        *offset = spdif_mpeg_pkt_offset[0][0];

        *codec = AV_CODEC_ID_MP1;

        break;

    case IEC61937_MPEG2_LAYER2_LSF:

        *offset = spdif_mpeg_pkt_offset[0][1];

        *codec = AV_CODEC_ID_MP2;

        break;

    case IEC61937_MPEG2_LAYER3_LSF:

        *offset = spdif_mpeg_pkt_offset[0][2];

        *codec = AV_CODEC_ID_MP3;

        break;

    case IEC61937_DTS1:

        *offset = 2048;

        *codec = AV_CODEC_ID_DTS;

        break;

    case IEC61937_DTS2:

        *offset = 4096;

        *codec = AV_CODEC_ID_DTS;

        break;

    case IEC61937_DTS3:

        *offset = 8192;

        *codec = AV_CODEC_ID_DTS;

        break;

    default:

        if (s) { /* be silent during a probe */

            avpriv_request_sample(s, "Data type 0x%04x in IEC 61937",

                                  data_type);

        }

        return AVERROR_PATCHWELCOME;

    }

    return 0;

}
