static void read_apic(AVFormatContext *s, AVIOContext *pb, int taglen, char *tag, ID3v2ExtraMeta **extra_meta)

{

    int enc, pic_type;

    char             mimetype[64];

    const CodecMime     *mime = ff_id3v2_mime_tags;

    enum AVCodecID           id = AV_CODEC_ID_NONE;

    ID3v2ExtraMetaAPIC  *apic = NULL;

    ID3v2ExtraMeta *new_extra = NULL;

    int64_t               end = avio_tell(pb) + taglen;



    if (taglen <= 4)

        goto fail;



    new_extra = av_mallocz(sizeof(*new_extra));

    apic      = av_mallocz(sizeof(*apic));

    if (!new_extra || !apic)

        goto fail;



    enc = avio_r8(pb);

    taglen--;



    /* mimetype */

    taglen -= avio_get_str(pb, taglen, mimetype, sizeof(mimetype));

    while (mime->id != AV_CODEC_ID_NONE) {

        if (!av_strncasecmp(mime->str, mimetype, sizeof(mimetype))) {

            id = mime->id;

            break;

        }

        mime++;

    }

    if (id == AV_CODEC_ID_NONE) {

        av_log(s, AV_LOG_WARNING, "Unknown attached picture mimetype: %s, skipping.\n", mimetype);

        goto fail;

    }

    apic->id = id;



    /* picture type */

    pic_type = avio_r8(pb);

    taglen--;

    if (pic_type < 0 || pic_type >= FF_ARRAY_ELEMS(ff_id3v2_picture_types)) {

        av_log(s, AV_LOG_WARNING, "Unknown attached picture type %d.\n", pic_type);

        pic_type = 0;

    }

    apic->type = ff_id3v2_picture_types[pic_type];



    /* description and picture data */

    if (decode_str(s, pb, enc, &apic->description, &taglen) < 0) {

        av_log(s, AV_LOG_ERROR, "Error decoding attached picture description.\n");

        goto fail;

    }



    apic->buf = av_buffer_alloc(taglen);

    if (!apic->buf || !taglen || avio_read(pb, apic->buf->data, taglen) != taglen)

        goto fail;



    new_extra->tag    = "APIC";

    new_extra->data   = apic;

    new_extra->next   = *extra_meta;

    *extra_meta       = new_extra;



    return;



fail:

    if (apic)

        free_apic(apic);

    av_freep(&new_extra);

    avio_seek(pb, end, SEEK_SET);

}
