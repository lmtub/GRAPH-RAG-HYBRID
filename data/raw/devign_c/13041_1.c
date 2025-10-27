static int parse_h264_sdp_line(AVFormatContext *s, int st_index,

                               PayloadContext *h264_data, const char *line)

{

    AVStream *stream;

    AVCodecContext *codec;

    const char *p = line;



    if (st_index < 0)

        return 0;



    stream = s->streams[st_index];

    codec = stream->codec;

    assert(h264_data->cookie == MAGIC_COOKIE);



    if (av_strstart(p, "framesize:", &p)) {

        char buf1[50];

        char *dst = buf1;



        // remove the protocol identifier..

        while (*p && *p == ' ') p++; // strip spaces.

        while (*p && *p != ' ') p++; // eat protocol identifier

        while (*p && *p == ' ') p++; // strip trailing spaces.

        while (*p && *p != '-' && (dst - buf1) < sizeof(buf1) - 1) {

            *dst++ = *p++;

        }

        *dst = '\0';



        // a='framesize:96 320-240'

        // set our parameters..

        codec->width = atoi(buf1);

        codec->height = atoi(p + 1); // skip the -

        codec->pix_fmt = PIX_FMT_YUV420P;

    } else if (av_strstart(p, "fmtp:", &p)) {

        return ff_parse_fmtp(stream, h264_data, p, sdp_parse_fmtp_config_h264);

    } else if (av_strstart(p, "cliprect:", &p)) {

        // could use this if we wanted.

    }



    return 0;                   // keep processing it the normal way...

}
