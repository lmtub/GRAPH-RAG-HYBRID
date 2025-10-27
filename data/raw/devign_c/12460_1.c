static int rm_read_header(AVFormatContext *s)

{

    RMDemuxContext *rm = s->priv_data;

    AVStream *st;

    AVIOContext *pb = s->pb;

    unsigned int tag;

    int tag_size;

    unsigned int start_time, duration;

    unsigned int data_off = 0, indx_off = 0;

    char buf[128], mime[128];

    int flags = 0;



    tag = avio_rl32(pb);

    if (tag == MKTAG('.', 'r', 'a', 0xfd)) {

        /* very old .ra format */

        return rm_read_header_old(s);

    } else if (tag != MKTAG('.', 'R', 'M', 'F')) {

        return AVERROR(EIO);

    }



    tag_size = avio_rb32(pb);

    avio_skip(pb, tag_size - 8);



    for(;;) {

        if (url_feof(pb))

            return -1;

        tag = avio_rl32(pb);

        tag_size = avio_rb32(pb);

        avio_rb16(pb);

        av_dlog(s, "tag=%c%c%c%c (%08x) size=%d\n",

                (tag      ) & 0xff,

                (tag >>  8) & 0xff,

                (tag >> 16) & 0xff,

                (tag >> 24) & 0xff,

                tag,

                tag_size);

        if (tag_size < 10 && tag != MKTAG('D', 'A', 'T', 'A'))

            return -1;

        switch(tag) {

        case MKTAG('P', 'R', 'O', 'P'):

            /* file header */

            avio_rb32(pb); /* max bit rate */

            avio_rb32(pb); /* avg bit rate */

            avio_rb32(pb); /* max packet size */

            avio_rb32(pb); /* avg packet size */

            avio_rb32(pb); /* nb packets */

            duration = avio_rb32(pb); /* duration */

            s->duration = av_rescale(duration, AV_TIME_BASE, 1000);

            avio_rb32(pb); /* preroll */

            indx_off = avio_rb32(pb); /* index offset */

            data_off = avio_rb32(pb); /* data offset */

            avio_rb16(pb); /* nb streams */

            flags = avio_rb16(pb); /* flags */

            break;

        case MKTAG('C', 'O', 'N', 'T'):

            rm_read_metadata(s, 1);

            break;

        case MKTAG('M', 'D', 'P', 'R'):

            st = avformat_new_stream(s, NULL);

            if (!st)

                return AVERROR(ENOMEM);

            st->id = avio_rb16(pb);

            avio_rb32(pb); /* max bit rate */

            st->codec->bit_rate = avio_rb32(pb); /* bit rate */

            avio_rb32(pb); /* max packet size */

            avio_rb32(pb); /* avg packet size */

            start_time = avio_rb32(pb); /* start time */

            avio_rb32(pb); /* preroll */

            duration = avio_rb32(pb); /* duration */

            st->start_time = start_time;

            st->duration = duration;

            if(duration>0)

                s->duration = AV_NOPTS_VALUE;

            get_str8(pb, buf, sizeof(buf)); /* desc */

            get_str8(pb, mime, sizeof(mime)); /* mimetype */

            st->codec->codec_type = AVMEDIA_TYPE_DATA;

            st->priv_data = ff_rm_alloc_rmstream();

            if (ff_rm_read_mdpr_codecdata(s, s->pb, st, st->priv_data,

                                          avio_rb32(pb), mime) < 0)

                return -1;

            break;

        case MKTAG('D', 'A', 'T', 'A'):

            goto header_end;

        default:

            /* unknown tag: skip it */

            avio_skip(pb, tag_size - 10);

            break;

        }

    }

 header_end:

    rm->nb_packets = avio_rb32(pb); /* number of packets */

    if (!rm->nb_packets && (flags & 4))

        rm->nb_packets = 3600 * 25;

    avio_rb32(pb); /* next data header */



    if (!data_off)

        data_off = avio_tell(pb) - 18;

    if (indx_off && pb->seekable && !(s->flags & AVFMT_FLAG_IGNIDX) &&

        avio_seek(pb, indx_off, SEEK_SET) >= 0) {

        rm_read_index(s);

        avio_seek(pb, data_off + 18, SEEK_SET);

    }



    return 0;

}
