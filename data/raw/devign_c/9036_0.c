static int flv_read_metabody(AVFormatContext *s, int64_t next_pos)

{

    AMFDataType type;

    AVStream *stream, *astream, *vstream;

    AVIOContext *ioc;

    int i;

    // only needs to hold the string "onMetaData".

    // Anything longer is something we don't want.

    char buffer[11];



    astream = NULL;

    vstream = NULL;

    ioc     = s->pb;



    // first object needs to be "onMetaData" string

    type = avio_r8(ioc);

    if (type != AMF_DATA_TYPE_STRING ||

        amf_get_string(ioc, buffer, sizeof(buffer)) < 0)

        return -1;



    if (!strcmp(buffer, "onTextData"))

        return 1;



    if (strcmp(buffer, "onMetaData"))

        return -1;



    // find the streams now so that amf_parse_object doesn't need to do

    // the lookup every time it is called.

    for (i = 0; i < s->nb_streams; i++) {

        stream = s->streams[i];

        if (stream->codec->codec_type == AVMEDIA_TYPE_AUDIO)

            astream = stream;

        else if (stream->codec->codec_type == AVMEDIA_TYPE_VIDEO)

            vstream = stream;

    }



    // parse the second object (we want a mixed array)

    if (amf_parse_object(s, astream, vstream, buffer, next_pos, 0) < 0)

        return -1;



    return 0;

}
