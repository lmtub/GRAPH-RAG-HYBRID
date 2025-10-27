static int write_representation(AVFormatContext *s, AVStream *stream, int id,

                                int output_width, int output_height,

                                int output_sample_rate) {

    AVDictionaryEntry *irange = av_dict_get(stream->metadata, INITIALIZATION_RANGE, NULL, 0);

    AVDictionaryEntry *cues_start = av_dict_get(stream->metadata, CUES_START, NULL, 0);

    AVDictionaryEntry *cues_end = av_dict_get(stream->metadata, CUES_END, NULL, 0);

    AVDictionaryEntry *filename = av_dict_get(stream->metadata, FILENAME, NULL, 0);

    AVDictionaryEntry *bandwidth = av_dict_get(stream->metadata, BANDWIDTH, NULL, 0);

    if (!irange || cues_start == NULL || cues_end == NULL || filename == NULL ||

        !bandwidth) {

        return -1;

    }

    avio_printf(s->pb, "<Representation id=\"%d\"", id);

    avio_printf(s->pb, " bandwidth=\"%s\"", bandwidth->value);

    if (stream->codec->codec_type == AVMEDIA_TYPE_VIDEO && output_width)

        avio_printf(s->pb, " width=\"%d\"", stream->codec->width);

    if (stream->codec->codec_type == AVMEDIA_TYPE_VIDEO && output_height)

        avio_printf(s->pb, " height=\"%d\"", stream->codec->height);

    if (stream->codec->codec_type = AVMEDIA_TYPE_AUDIO && output_sample_rate)

        avio_printf(s->pb, " audioSamplingRate=\"%d\"", stream->codec->sample_rate);

    avio_printf(s->pb, ">\n");

    avio_printf(s->pb, "<BaseURL>%s</BaseURL>\n", filename->value);

    avio_printf(s->pb, "<SegmentBase\n");

    avio_printf(s->pb, "  indexRange=\"%s-%s\">\n", cues_start->value, cues_end->value);

    avio_printf(s->pb, "<Initialization\n");

    avio_printf(s->pb, "  range=\"0-%s\" />\n", irange->value);

    avio_printf(s->pb, "</SegmentBase>\n");

    avio_printf(s->pb, "</Representation>\n");

    return 0;

}
