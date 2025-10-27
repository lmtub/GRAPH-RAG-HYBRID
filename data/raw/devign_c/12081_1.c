static int swf_read_header(AVFormatContext *s, AVFormatParameters *ap)

{

    ByteIOContext *pb = &s->pb;

    int nbits, len, frame_rate, tag, v;

    AVStream *st;



    if ((get_be32(pb) & 0xffffff00) != MKBETAG('F', 'W', 'S', 0))

        return -EIO;

    get_le32(pb);

    /* skip rectangle size */

    nbits = get_byte(pb) >> 3;

    len = (4 * nbits - 3 + 7) / 8;

    url_fskip(pb, len);

    frame_rate = get_le16(pb);

    get_le16(pb); /* frame count */



    for(;;) {

        tag = get_swf_tag(pb, &len);

        if (tag < 0) {

            fprintf(stderr, "No streaming found in SWF\n");

            return -EIO;

        }

        if (tag == TAG_STREAMHEAD) {

            /* streaming found */

            get_byte(pb);

            v = get_byte(pb);

            get_le16(pb);

            if (len!=4)

                url_fskip(pb,len-4);

            /* if mp3 streaming found, OK */

            if ((v & 0x20) != 0) {

                st = av_new_stream(s, 0);

                if (!st)

                    return -ENOMEM;



                if (v & 0x01)

                    st->codec.channels = 2;

                else

                    st->codec.channels = 1;



                switch((v>> 2) & 0x03) {

                case 1:

                    st->codec.sample_rate = 11025;

                    break;

                case 2:

                    st->codec.sample_rate = 22050;

                    break;

                case 3:

                    st->codec.sample_rate = 44100;

                    break;

                default:

                    av_free(st);

                    return -EIO;

                }

                st->codec.codec_type = CODEC_TYPE_AUDIO;

                st->codec.codec_id = CODEC_ID_MP2;

                break;

            }

        } else {

            url_fskip(pb, len);

        }

    }



    return 0;

}
