static int ape_read_header(AVFormatContext * s)

{

    AVIOContext *pb = s->pb;

    APEContext *ape = s->priv_data;

    AVStream *st;

    uint32_t tag;

    int i;

    int total_blocks, final_size = 0;

    int64_t pts, file_size;



    /* Skip any leading junk such as id3v2 tags */

    ape->junklength = avio_tell(pb);



    tag = avio_rl32(pb);

    if (tag != MKTAG('M', 'A', 'C', ' '))

        return -1;



    ape->fileversion = avio_rl16(pb);



    if (ape->fileversion < APE_MIN_VERSION || ape->fileversion > APE_MAX_VERSION) {

        av_log(s, AV_LOG_ERROR, "Unsupported file version - %d.%02d\n",

               ape->fileversion / 1000, (ape->fileversion % 1000) / 10);

        return -1;

    }



    if (ape->fileversion >= 3980) {

        ape->padding1             = avio_rl16(pb);

        ape->descriptorlength     = avio_rl32(pb);

        ape->headerlength         = avio_rl32(pb);

        ape->seektablelength      = avio_rl32(pb);

        ape->wavheaderlength      = avio_rl32(pb);

        ape->audiodatalength      = avio_rl32(pb);

        ape->audiodatalength_high = avio_rl32(pb);

        ape->wavtaillength        = avio_rl32(pb);

        avio_read(pb, ape->md5, 16);



        /* Skip any unknown bytes at the end of the descriptor.

           This is for future compatibility */

        if (ape->descriptorlength > 52)

            avio_skip(pb, ape->descriptorlength - 52);



        /* Read header data */

        ape->compressiontype      = avio_rl16(pb);

        ape->formatflags          = avio_rl16(pb);

        ape->blocksperframe       = avio_rl32(pb);

        ape->finalframeblocks     = avio_rl32(pb);

        ape->totalframes          = avio_rl32(pb);

        ape->bps                  = avio_rl16(pb);

        ape->channels             = avio_rl16(pb);

        ape->samplerate           = avio_rl32(pb);

    } else {

        ape->descriptorlength = 0;

        ape->headerlength = 32;



        ape->compressiontype      = avio_rl16(pb);

        ape->formatflags          = avio_rl16(pb);

        ape->channels             = avio_rl16(pb);

        ape->samplerate           = avio_rl32(pb);

        ape->wavheaderlength      = avio_rl32(pb);

        ape->wavtaillength        = avio_rl32(pb);

        ape->totalframes          = avio_rl32(pb);

        ape->finalframeblocks     = avio_rl32(pb);



        if (ape->formatflags & MAC_FORMAT_FLAG_HAS_PEAK_LEVEL) {

            avio_skip(pb, 4); /* Skip the peak level */

            ape->headerlength += 4;

        }



        if (ape->formatflags & MAC_FORMAT_FLAG_HAS_SEEK_ELEMENTS) {

            ape->seektablelength = avio_rl32(pb);

            ape->headerlength += 4;

            ape->seektablelength *= sizeof(int32_t);

        } else

            ape->seektablelength = ape->totalframes * sizeof(int32_t);



        if (ape->formatflags & MAC_FORMAT_FLAG_8_BIT)

            ape->bps = 8;

        else if (ape->formatflags & MAC_FORMAT_FLAG_24_BIT)

            ape->bps = 24;

        else

            ape->bps = 16;



        if (ape->fileversion >= 3950)

            ape->blocksperframe = 73728 * 4;

        else if (ape->fileversion >= 3900 || (ape->fileversion >= 3800  && ape->compressiontype >= 4000))

            ape->blocksperframe = 73728;

        else

            ape->blocksperframe = 9216;



        /* Skip any stored wav header */

        if (!(ape->formatflags & MAC_FORMAT_FLAG_CREATE_WAV_HEADER))

            avio_skip(pb, ape->wavheaderlength);

    }



    if(!ape->totalframes){

        av_log(s, AV_LOG_ERROR, "No frames in the file!\n");

        return AVERROR(EINVAL);

    }

    if(ape->totalframes > UINT_MAX / sizeof(APEFrame)){

        av_log(s, AV_LOG_ERROR, "Too many frames: %"PRIu32"\n",

               ape->totalframes);

        return -1;

    }

    if (ape->seektablelength / sizeof(*ape->seektable) < ape->totalframes) {

        av_log(s, AV_LOG_ERROR,

               "Number of seek entries is less than number of frames: %zu vs. %"PRIu32"\n",

               ape->seektablelength / sizeof(*ape->seektable), ape->totalframes);

        return AVERROR_INVALIDDATA;

    }

    ape->frames       = av_malloc(ape->totalframes * sizeof(APEFrame));

    if(!ape->frames)

        return AVERROR(ENOMEM);

    ape->firstframe   = ape->junklength + ape->descriptorlength + ape->headerlength + ape->seektablelength + ape->wavheaderlength;

    if (ape->fileversion < 3810)

        ape->firstframe += ape->totalframes;

    ape->currentframe = 0;





    ape->totalsamples = ape->finalframeblocks;

    if (ape->totalframes > 1)

        ape->totalsamples += ape->blocksperframe * (ape->totalframes - 1);



    if (ape->seektablelength > 0) {

        ape->seektable = av_malloc(ape->seektablelength);

        if (!ape->seektable)

            return AVERROR(ENOMEM);

        for (i = 0; i < ape->seektablelength / sizeof(uint32_t) && !pb->eof_reached; i++)

            ape->seektable[i] = avio_rl32(pb);

        if (ape->fileversion < 3810) {

            ape->bittable = av_malloc(ape->totalframes);

            if (!ape->bittable)

                return AVERROR(ENOMEM);

            for (i = 0; i < ape->totalframes && !pb->eof_reached; i++)

                ape->bittable[i] = avio_r8(pb);

        }

    }



    ape->frames[0].pos     = ape->firstframe;

    ape->frames[0].nblocks = ape->blocksperframe;

    ape->frames[0].skip    = 0;

    for (i = 1; i < ape->totalframes; i++) {

        ape->frames[i].pos      = ape->seektable[i] + ape->junklength;

        ape->frames[i].nblocks  = ape->blocksperframe;

        ape->frames[i - 1].size = ape->frames[i].pos - ape->frames[i - 1].pos;

        ape->frames[i].skip     = (ape->frames[i].pos - ape->frames[0].pos) & 3;

    }

    ape->frames[ape->totalframes - 1].nblocks = ape->finalframeblocks;

    /* calculate final packet size from total file size, if available */

    file_size = avio_size(pb);

    if (file_size > 0) {

        final_size = file_size - ape->frames[ape->totalframes - 1].pos -

                     ape->wavtaillength;

        final_size -= final_size & 3;

    }

    if (file_size <= 0 || final_size <= 0)

        final_size = ape->finalframeblocks * 8;

    ape->frames[ape->totalframes - 1].size = final_size;



    for (i = 0; i < ape->totalframes; i++) {

        if(ape->frames[i].skip){

            ape->frames[i].pos  -= ape->frames[i].skip;

            ape->frames[i].size += ape->frames[i].skip;

        }

        ape->frames[i].size = (ape->frames[i].size + 3) & ~3;

    }

    if (ape->fileversion < 3810) {

        for (i = 0; i < ape->totalframes; i++) {

            if (i < ape->totalframes - 1 && ape->bittable[i + 1])

                ape->frames[i].size += 4;

            ape->frames[i].skip <<= 3;

            ape->frames[i].skip  += ape->bittable[i];

        }

    }



    ape_dumpinfo(s, ape);



    av_log(s, AV_LOG_DEBUG, "Decoding file - v%d.%02d, compression level %"PRIu16"\n",

           ape->fileversion / 1000, (ape->fileversion % 1000) / 10,

           ape->compressiontype);



    /* now we are ready: build format streams */

    st = avformat_new_stream(s, NULL);

    if (!st)

        return -1;



    total_blocks = (ape->totalframes == 0) ? 0 : ((ape->totalframes - 1) * ape->blocksperframe) + ape->finalframeblocks;



    st->codecpar->codec_type      = AVMEDIA_TYPE_AUDIO;

    st->codecpar->codec_id        = AV_CODEC_ID_APE;

    st->codecpar->codec_tag       = MKTAG('A', 'P', 'E', ' ');

    st->codecpar->channels        = ape->channels;

    st->codecpar->sample_rate     = ape->samplerate;

    st->codecpar->bits_per_coded_sample = ape->bps;



    st->nb_frames = ape->totalframes;

    st->start_time = 0;

    st->duration  = total_blocks / MAC_SUBFRAME_SIZE;

    avpriv_set_pts_info(st, 64, MAC_SUBFRAME_SIZE, ape->samplerate);



    st->codecpar->extradata = av_malloc(APE_EXTRADATA_SIZE);

    st->codecpar->extradata_size = APE_EXTRADATA_SIZE;

    AV_WL16(st->codecpar->extradata + 0, ape->fileversion);

    AV_WL16(st->codecpar->extradata + 2, ape->compressiontype);

    AV_WL16(st->codecpar->extradata + 4, ape->formatflags);



    pts = 0;

    for (i = 0; i < ape->totalframes; i++) {

        ape->frames[i].pts = pts;

        av_add_index_entry(st, ape->frames[i].pos, ape->frames[i].pts, 0, 0, AVINDEX_KEYFRAME);

        pts += ape->blocksperframe / MAC_SUBFRAME_SIZE;

    }



    /* try to read APE tags */

    if (pb->seekable) {

        ff_ape_parse_tag(s);

        avio_seek(pb, 0, SEEK_SET);

    }



    return 0;

}
