static int mxf_parse_mpeg2_frame(AVFormatContext *s, AVStream *st, AVPacket *pkt, int *flags)

{

    MXFStreamContext *sc = st->priv_data;

    MXFContext *mxf = s->priv_data;

    uint32_t c = -1;

    int i;



    *flags = 0;



    for(i = 0; i < pkt->size - 4; i++) {

        c = (c<<8) + pkt->data[i];

        if (c == 0x1B5) {

            if (i + 2 < pkt->size && (pkt->data[i+1] & 0xf0) == 0x10) { // seq ext

                st->codec->profile = pkt->data[i+1] & 0x07;

                st->codec->level   = pkt->data[i+2] >> 4;

            } else if (i + 5 < pkt->size && (pkt->data[i+1] & 0xf0) == 0x80) { // pict coding ext

                sc->interlaced = !(pkt->data[i+5] & 0x80); // progressive frame

                break;

            }

        } else if (c == 0x1b8) { // gop

            if (i + 4 < pkt->size) {

                if (pkt->data[i+4]>>6 & 0x01) // closed

                    *flags |= 0x80; // random access

                if (!mxf->header_written) {

                    unsigned hours   =  (pkt->data[i+1]>>2) & 0x1f;

                    unsigned minutes = ((pkt->data[i+1] & 0x03) << 4) | (pkt->data[i+2]>>4);

                    unsigned seconds = ((pkt->data[i+2] & 0x07) << 3) | (pkt->data[i+3]>>5);

                    unsigned frames  = ((pkt->data[i+3] & 0x1f) << 1) | (pkt->data[i+4]>>7);

                    mxf->timecode_drop_frame = !!(pkt->data[i+1] & 0x80);

                    mxf->timecode_start = (hours*3600 + minutes*60 + seconds) *

                        mxf->timecode_base + frames;

                    if (mxf->timecode_drop_frame) {

                        unsigned tminutes = 60 * hours + minutes;

                        mxf->timecode_start -= 2 * (tminutes - tminutes / 10);

                    }

                    av_log(s, AV_LOG_DEBUG, "frame %d %d:%d:%d%c%d\n", mxf->timecode_start,

                           hours, minutes, seconds, mxf->timecode_drop_frame ? ';':':', frames);

                }

            }

        } else if (c == 0x1b3) { // seq

            *flags |= 0x40;

            if (i + 4 < pkt->size) {

                switch ((pkt->data[i+4]>>4) & 0xf) {

                case 2:  sc->aspect_ratio = (AVRational){  4,  3}; break;

                case 3:  sc->aspect_ratio = (AVRational){ 16,  9}; break;

                case 4:  sc->aspect_ratio = (AVRational){221,100}; break;

                default:

                    av_reduce(&sc->aspect_ratio.num, &sc->aspect_ratio.den,

                              st->codec->width, st->codec->height, 1024*1024);

                }

            }

        } else if (c == 0x100) { // pic

            int pict_type = (pkt->data[i+2]>>3) & 0x07;

            if (pict_type == 2) { // P frame

                *flags |= 0x22;

                st->codec->gop_size = 1;

            } else if (pict_type == 3) { // B frame

                *flags |= 0x33;

                sc->temporal_reordering = -1;

            } else if (!pict_type) {

                av_log(s, AV_LOG_ERROR, "error parsing mpeg2 frame\n");

                return 0;

            }

        }

    }

    if (s->oformat != &mxf_d10_muxer)

        sc->codec_ul = mxf_get_mpeg2_codec_ul(st->codec);

    return !!sc->codec_ul;

}
