static int aiff_read_header(AVFormatContext *s)

{

    int size, filesize;

    int64_t offset = 0;

    uint32_t tag;

    unsigned version = AIFF_C_VERSION1;

    AVIOContext *pb = s->pb;

    AVStream * st;

    AIFFInputContext *aiff = s->priv_data;



    /* check FORM header */

    filesize = get_tag(pb, &tag);

    if (filesize < 0 || tag != MKTAG('F', 'O', 'R', 'M'))

        return AVERROR_INVALIDDATA;



    /* AIFF data type */

    tag = avio_rl32(pb);

    if (tag == MKTAG('A', 'I', 'F', 'F'))       /* Got an AIFF file */

        version = AIFF;

    else if (tag != MKTAG('A', 'I', 'F', 'C'))  /* An AIFF-C file then */

        return AVERROR_INVALIDDATA;



    filesize -= 4;



    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);



    while (filesize > 0) {

        /* parse different chunks */

        size = get_tag(pb, &tag);

        if (size < 0)

            return size;



        filesize -= size + 8;



        switch (tag) {

        case MKTAG('C', 'O', 'M', 'M'):     /* Common chunk */

            /* Then for the complete header info */

            st->nb_frames = get_aiff_header(s, size, version);

            if (st->nb_frames < 0)

                return st->nb_frames;

            if (offset > 0) // COMM is after SSND

                goto got_sound;

            break;

        case MKTAG('F', 'V', 'E', 'R'):     /* Version chunk */

            version = avio_rb32(pb);

            break;

        case MKTAG('N', 'A', 'M', 'E'):     /* Sample name chunk */

            get_meta(s, "title"    , size);

            break;

        case MKTAG('A', 'U', 'T', 'H'):     /* Author chunk */

            get_meta(s, "author"   , size);

            break;

        case MKTAG('(', 'c', ')', ' '):     /* Copyright chunk */

            get_meta(s, "copyright", size);

            break;

        case MKTAG('A', 'N', 'N', 'O'):     /* Annotation chunk */

            get_meta(s, "comment"  , size);

            break;

        case MKTAG('S', 'S', 'N', 'D'):     /* Sampled sound chunk */

            aiff->data_end = avio_tell(pb) + size;

            offset = avio_rb32(pb);      /* Offset of sound data */

            avio_rb32(pb);               /* BlockSize... don't care */

            offset += avio_tell(pb);    /* Compute absolute data offset */

            if (st->codecpar->block_align)    /* Assume COMM already parsed */

                goto got_sound;

            if (!pb->seekable) {

                av_log(s, AV_LOG_ERROR, "file is not seekable\n");

                return -1;

            }

            avio_skip(pb, size - 8);

            break;

        case MKTAG('w', 'a', 'v', 'e'):

            if ((uint64_t)size > (1<<30))

                return -1;

            st->codecpar->extradata = av_mallocz(size + AV_INPUT_BUFFER_PADDING_SIZE);

            if (!st->codecpar->extradata)

                return AVERROR(ENOMEM);

            st->codecpar->extradata_size = size;

            avio_read(pb, st->codecpar->extradata, size);

            break;

        default: /* Jump */

            avio_skip(pb, size);

        }



        /* Skip required padding byte for odd-sized chunks. */

        if (size & 1) {

            filesize--;

            avio_skip(pb, 1);

        }

    }



got_sound:

    if (!st->codecpar->block_align) {

        av_log(s, AV_LOG_ERROR, "could not find COMM tag or invalid block_align value\n");

        return -1;

    }



    /* Now positioned, get the sound data start and end */

    avpriv_set_pts_info(st, 64, 1, st->codecpar->sample_rate);

    st->start_time = 0;

    st->duration = st->nb_frames * aiff->block_duration;



    /* Position the stream at the first block */

    avio_seek(pb, offset, SEEK_SET);



    return 0;

}
