static void write_codec_attr(AVStream *st, VariantStream *vs) {

    int codec_strlen = strlen(vs->codec_attr);

    char attr[32];



    if (st->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE)

        return;

    if (vs->attr_status == CODEC_ATTRIBUTE_WILL_NOT_BE_WRITTEN)

        return;



    if (st->codecpar->codec_id == AV_CODEC_ID_H264) {

        uint8_t *data = st->codecpar->extradata;

        if ((data[0] | data[1] | data[2]) == 0 && data[3] == 1 && (data[4] & 0x1F) == 7) {

            snprintf(attr, sizeof(attr),

                     "avc1.%02x%02x%02x", data[5], data[6], data[7]);

        } else {

            goto fail;

        }

    } else if (st->codecpar->codec_id == AV_CODEC_ID_MP2) {

        snprintf(attr, sizeof(attr), "mp4a.40.33");

    } else if (st->codecpar->codec_id == AV_CODEC_ID_MP3) {

        snprintf(attr, sizeof(attr), "mp4a.40.34");

    } else if (st->codecpar->codec_id == AV_CODEC_ID_AAC) {

        /* TODO : For HE-AAC, HE-AACv2, the last digit needs to be set to 5 and 29 respectively */

        snprintf(attr, sizeof(attr), "mp4a.40.2");

    } else if (st->codecpar->codec_id == AV_CODEC_ID_AC3) {

        snprintf(attr, sizeof(attr), "ac-3");

    } else if (st->codecpar->codec_id == AV_CODEC_ID_EAC3) {

        snprintf(attr, sizeof(attr), "ec-3");

    } else {

        goto fail;

    }

    // Don't write the same attribute multiple times

    if (!av_stristr(vs->codec_attr, attr)) {

        snprintf(vs->codec_attr + codec_strlen,

                 sizeof(vs->codec_attr) - codec_strlen,

                 "%s%s", codec_strlen ? "," : "", attr);

    }

    return;



fail:

    vs->codec_attr[0] = '\0';

    vs->attr_status = CODEC_ATTRIBUTE_WILL_NOT_BE_WRITTEN;

    return;

}
