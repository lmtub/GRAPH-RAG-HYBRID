static int av_encode(AVFormatContext **output_files,
                     int nb_output_files,
                     AVFormatContext **input_files,
                     int nb_input_files,
                     AVStreamMap *stream_maps, int nb_stream_maps)
{
    int ret, i, j, k, n, nb_istreams = 0, nb_ostreams = 0, pts_set;
    AVFormatContext *is, *os;
    AVCodecContext *codec, *icodec;
    AVOutputStream *ost, **ost_table = NULL;
    AVInputStream *ist, **ist_table = NULL;
    AVInputFile *file_table;
    AVFormatContext *stream_no_data;
    int key;
    file_table= (AVInputFile*) av_mallocz(nb_input_files * sizeof(AVInputFile));
    if (!file_table)
        goto fail;
    /* input stream init */
    j = 0;
    for(i=0;i<nb_input_files;i++) {
        is = input_files[i];
        file_table[i].ist_index = j;
        file_table[i].nb_streams = is->nb_streams;
        j += is->nb_streams;
    nb_istreams = j;
    ist_table = av_mallocz(nb_istreams * sizeof(AVInputStream *));
    if (!ist_table)
        goto fail;
    for(i=0;i<nb_istreams;i++) {
        ist = av_mallocz(sizeof(AVInputStream));
        if (!ist)
            goto fail;
        ist_table[i] = ist;
    j = 0;
    for(i=0;i<nb_input_files;i++) {
        is = input_files[i];
        for(k=0;k<is->nb_streams;k++) {
            ist = ist_table[j++];
            ist->st = is->streams[k];
            ist->file_index = i;
            ist->index = k;
            ist->discard = 1; /* the stream is discarded by default
                                 (changed later) */
            if (ist->st->codec.rate_emu) {
                ist->start = av_gettime();
                ist->frame = 0;
    /* output stream init */
    nb_ostreams = 0;
    for(i=0;i<nb_output_files;i++) {
        os = output_files[i];
        nb_ostreams += os->nb_streams;
    if (nb_stream_maps > 0 && nb_stream_maps != nb_ostreams) {
        fprintf(stderr, "Number of stream maps must match number of output streams\n");
    /* Sanity check the mapping args -- do the input files & streams exist? */
    for(i=0;i<nb_stream_maps;i++) {
        int fi = stream_maps[i].file_index;
        int si = stream_maps[i].stream_index;
        if (fi < 0 || fi > nb_input_files - 1 ||
            si < 0 || si > file_table[fi].nb_streams - 1) {
            fprintf(stderr,"Could not find input stream #%d.%d\n", fi, si);
    ost_table = av_mallocz(sizeof(AVOutputStream *) * nb_ostreams);
    if (!ost_table)
        goto fail;
    for(i=0;i<nb_ostreams;i++) {
        ost = av_mallocz(sizeof(AVOutputStream));
        if (!ost)
            goto fail;
        ost_table[i] = ost;
    n = 0;
    for(k=0;k<nb_output_files;k++) {
        os = output_files[k];
        for(i=0;i<os->nb_streams;i++) {
            int found;
            ost = ost_table[n++];
            ost->file_index = k;
            ost->index = i;
            ost->st = os->streams[i];
            if (nb_stream_maps > 0) {
                ost->source_index = file_table[stream_maps[n-1].file_index].ist_index + 
                    stream_maps[n-1].stream_index;
            } else {
                /* get corresponding input stream index : we select the first one with the right type */
                found = 0;
                for(j=0;j<nb_istreams;j++) {
                    ist = ist_table[j];
                    if (ist->discard && 
                        ist->st->codec.codec_type == ost->st->codec.codec_type) {
                        ost->source_index = j;
                        found = 1;
                if (!found) {
                    /* try again and reuse existing stream */
                    for(j=0;j<nb_istreams;j++) {
                        ist = ist_table[j];
                        if (ist->st->codec.codec_type == ost->st->codec.codec_type) {
                            ost->source_index = j;
                            found = 1;
                    if (!found) {
                        fprintf(stderr, "Could not find input stream matching output stream #%d.%d\n",
            ist = ist_table[ost->source_index];
            ist->discard = 0;
    /* for each output stream, we compute the right encoding parameters */
    for(i=0;i<nb_ostreams;i++) {
        ost = ost_table[i];
        ist = ist_table[ost->source_index];
        codec = &ost->st->codec;
        icodec = &ist->st->codec;
        if (ost->st->stream_copy) {
            /* if stream_copy is selected, no need to decode or encode */
            codec->codec_id = icodec->codec_id;
            codec->codec_type = icodec->codec_type;
            codec->codec_tag = icodec->codec_tag;
            codec->bit_rate = icodec->bit_rate;
            switch(codec->codec_type) {
            case CODEC_TYPE_AUDIO:
                codec->sample_rate = icodec->sample_rate;
                codec->channels = icodec->channels;
                break;
            case CODEC_TYPE_VIDEO:
                codec->frame_rate = icodec->frame_rate;
                codec->frame_rate_base = icodec->frame_rate_base;
                codec->width = icodec->width;
                codec->height = icodec->height;
                break;
            default:
                av_abort();
        } else {
            switch(codec->codec_type) {
            case CODEC_TYPE_AUDIO:
                if (fifo_init(&ost->fifo, 2 * MAX_AUDIO_PACKET_SIZE))
                    goto fail;
                if (codec->channels == icodec->channels &&
                    codec->sample_rate == icodec->sample_rate) {
                    ost->audio_resample = 0;
                } else {
                    if (codec->channels != icodec->channels &&
                        icodec->codec_id == CODEC_ID_AC3) {
                        /* Special case for 5:1 AC3 input */
                        /* and mono or stereo output      */
                        /* Request specific number of channels */
                        icodec->channels = codec->channels;
                        if (codec->sample_rate == icodec->sample_rate)
                            ost->audio_resample = 0;
                        else {
                            ost->audio_resample = 1;
                            ost->resample = audio_resample_init(codec->channels, icodec->channels,
                                                        codec->sample_rate, 
                                                        icodec->sample_rate);
                        /* Request specific number of channels */
                        icodec->channels = codec->channels;
                    } else {
                        ost->audio_resample = 1; 
                        ost->resample = audio_resample_init(codec->channels, icodec->channels,
                                                        codec->sample_rate, 
                                                        icodec->sample_rate);
                ist->decoding_needed = 1;
                ost->encoding_needed = 1;
                break;
            case CODEC_TYPE_VIDEO:
                if (codec->width == icodec->width &&
                    codec->height == icodec->height &&
                    frame_topBand == 0 &&
                    frame_bottomBand == 0 &&
                    frame_leftBand == 0 &&
                    frame_rightBand == 0)
                {
                    ost->video_resample = 0;
                    ost->video_crop = 0;
                } else if ((codec->width == icodec->width -
                                (frame_leftBand + frame_rightBand)) &&
                        (codec->height == icodec->height -
                                (frame_topBand  + frame_bottomBand)))
                {
                    ost->video_resample = 0;
                    ost->video_crop = 1;
                    ost->topBand = frame_topBand;
                    ost->leftBand = frame_leftBand;
                } else {
                    uint8_t *buf;
                    ost->video_resample = 1;
                    ost->video_crop = 0; // cropping is handled as part of resample
                    buf = av_malloc((codec->width * codec->height * 3) / 2);
                    if (!buf)
                        goto fail;
                    ost->pict_tmp.data[0] = buf;
                    ost->pict_tmp.data[1] = ost->pict_tmp.data[0] + (codec->width * codec->height);
                    ost->pict_tmp.data[2] = ost->pict_tmp.data[1] + (codec->width * codec->height) / 4;
                    ost->pict_tmp.linesize[0] = codec->width;
                    ost->pict_tmp.linesize[1] = codec->width / 2;
                    ost->pict_tmp.linesize[2] = codec->width / 2;
                    ost->img_resample_ctx = img_resample_full_init( 
                                      ost->st->codec.width, ost->st->codec.height,
                                      ist->st->codec.width, ist->st->codec.height,
                                      frame_topBand, frame_bottomBand,
                                      frame_leftBand, frame_rightBand);
                ost->encoding_needed = 1;
                ist->decoding_needed = 1;
                break;
            default:
                av_abort();
            /* two pass mode */
            if (ost->encoding_needed && 
                (codec->flags & (CODEC_FLAG_PASS1 | CODEC_FLAG_PASS2))) {
                char logfilename[1024];
                FILE *f;
                int size;
                char *logbuffer;
                snprintf(logfilename, sizeof(logfilename), "%s-%d.log", 
                         pass_logfilename ? 
                         pass_logfilename : DEFAULT_PASS_LOGFILENAME, i);
                if (codec->flags & CODEC_FLAG_PASS1) {
                    f = fopen(logfilename, "w");
                    if (!f) {
                        perror(logfilename);
                    ost->logfile = f;
                } else {
                    /* read the log file */
                    f = fopen(logfilename, "r");
                    if (!f) {
                        perror(logfilename);
                    fseek(f, 0, SEEK_END);
                    size = ftell(f);
                    fseek(f, 0, SEEK_SET);
                    logbuffer = av_malloc(size + 1);
                    if (!logbuffer) {
                        fprintf(stderr, "Could not allocate log buffer\n");
                    fread(logbuffer, 1, size, f);
                    fclose(f);
                    logbuffer[size] = '\0';
                    codec->stats_in = logbuffer;
    /* dump the file output parameters - cannot be done before in case
       of stream copy */
    for(i=0;i<nb_output_files;i++) {
        dump_format(output_files[i], i, output_files[i]->filename, 1);
    /* dump the stream mapping */
    fprintf(stderr, "Stream mapping:\n");
    for(i=0;i<nb_ostreams;i++) {
        ost = ost_table[i];
        fprintf(stderr, "  Stream #%d.%d -> #%d.%d\n",
                ist_table[ost->source_index]->file_index,
                ist_table[ost->source_index]->index,
                ost->file_index, 
                ost->index);
    /* open each encoder */
    for(i=0;i<nb_ostreams;i++) {
        ost = ost_table[i];
        if (ost->encoding_needed) {
            AVCodec *codec;
            codec = avcodec_find_encoder(ost->st->codec.codec_id);
            if (!codec) {
                fprintf(stderr, "Unsupported codec for output stream #%d.%d\n", 
            if (avcodec_open(&ost->st->codec, codec) < 0) {
                fprintf(stderr, "Error while opening codec for stream #%d.%d - maybe incorrect parameters such as bit_rate, rate, width or height\n", 
    /* open each decoder */
    for(i=0;i<nb_istreams;i++) {
        ist = ist_table[i];
        if (ist->decoding_needed) {
            AVCodec *codec;
            codec = avcodec_find_decoder(ist->st->codec.codec_id);
            if (!codec) {
                fprintf(stderr, "Unsupported codec (id=%d) for input stream #%d.%d\n", 
                        ist->st->codec.codec_id, ist->file_index, ist->index);
            if (avcodec_open(&ist->st->codec, codec) < 0) {
                fprintf(stderr, "Error while opening codec for input stream #%d.%d\n", 
                        ist->file_index, ist->index);
            //if (ist->st->codec.codec_type == CODEC_TYPE_VIDEO)
            //    ist->st->codec.flags |= CODEC_FLAG_REPEAT_FIELD;
            ist->frame_decoded = 1;
    /* init pts */
    for(i=0;i<nb_istreams;i++) {
        ist = ist_table[i];
    /* compute buffer size max (should use a complete heuristic) */
    for(i=0;i<nb_input_files;i++) {
        file_table[i].buffer_size_max = 2048;
    /* open files and write file headers */
    for(i=0;i<nb_output_files;i++) {
        os = output_files[i];
        if (av_write_header(os) < 0) {
            fprintf(stderr, "Could not write header for output file #%d (incorrect codec paramters ?)\n", i);
            ret = -EINVAL;
            goto fail;
#ifndef CONFIG_WIN32
    if (!do_play) {
        fprintf(stderr, "Press [q] to stop encoding\n");
    } else {
        fprintf(stderr, "Press [q] to stop playing\n");
#endif
    term_init();
    stream_no_data = 0;
    key = -1;
    for(;;) {
        int file_index, ist_index;
        AVPacket pkt;
        uint8_t *ptr;
        int len;
        uint8_t *data_buf;
        int data_size, got_picture;
        AVPicture picture;
        short samples[AVCODEC_MAX_AUDIO_FRAME_SIZE / 2];
        void *buffer_to_free;
        double pts_min;
    redo:
        /* if 'q' pressed, exits */
        if (key) {
            /* read_key() returns 0 on EOF */
            key = read_key();
            if (key == 'q')
                break;
        /* select the stream that we must read now by looking at the
           smallest output pts */
        file_index = -1;
        pts_min = 1e10;
        for(i=0;i<nb_ostreams;i++) {
            double pts;
            ost = ost_table[i];
            os = output_files[ost->file_index];
            ist = ist_table[ost->source_index];
            pts = (double)ost->st->pts.val * os->pts_num / os->pts_den;
            if (!file_table[ist->file_index].eof_reached && 
                pts < pts_min) {
                pts_min = pts;
                file_index = ist->file_index;
        /* if none, if is finished */
        if (file_index < 0) {
            break;
        /* finish if recording time exhausted */
        if (recording_time > 0 && pts_min >= (recording_time / 1000000.0))
            break;
        /* read a packet from it and output it in the fifo */
        is = input_files[file_index];
        if (av_read_packet(is, &pkt) < 0) {
            file_table[file_index].eof_reached = 1;
            continue;
        if (!pkt.size) {
            stream_no_data = is;
        } else {
            stream_no_data = 0;
        if (do_hex_dump) {
            printf("stream #%d, size=%d:\n", pkt.stream_index, pkt.size);
            av_hex_dump(pkt.data, pkt.size);
        /* the following test is needed in case new streams appear
           dynamically in stream : we ignore them */
        if (pkt.stream_index >= file_table[file_index].nb_streams)
            goto discard_packet;
        ist_index = file_table[file_index].ist_index + pkt.stream_index;
        ist = ist_table[ist_index];
        if (ist->discard)
            goto discard_packet;
        // printf("read #%d.%d size=%d\n", ist->file_index, ist->index, pkt.size);
        len = pkt.size;
        ptr = pkt.data;
        pts_set = 0;
        while (len > 0) {
            int64_t ipts;
            ipts = AV_NOPTS_VALUE;
            /* decode the packet if needed */
            data_buf = NULL; /* fail safe */
            data_size = 0;
            if (ist->decoding_needed) {
                /* NOTE1: we only take into account the PTS if a new
                   frame has begun (MPEG semantics) */
                /* NOTE2: even if the fraction is not initialized,
                   av_frac_set can be used to set the integer part */
                if (ist->frame_decoded && 
                    pkt.pts != AV_NOPTS_VALUE && 
                    !pts_set) {
                    ipts = pkt.pts;
                    ist->frame_decoded = 0;
                    pts_set = 1;
                switch(ist->st->codec.codec_type) {
                case CODEC_TYPE_AUDIO:
                    /* XXX: could avoid copy if PCM 16 bits with same
                       endianness as CPU */
                    ret = avcodec_decode_audio(&ist->st->codec, samples, &data_size,
                                               ptr, len);
                    if (ret < 0)
                        goto fail_decode;
                    /* Some bug in mpeg audio decoder gives */
                    /* data_size < 0, it seems they are overflows */
                    if (data_size <= 0) {
                        /* no audio frame */
                        ptr += ret;
                        len -= ret;
                        continue;
                    data_buf = (uint8_t *)samples;
                    break;
                case CODEC_TYPE_VIDEO:
                    {
                        AVFrame big_picture;
                        data_size = (ist->st->codec.width * ist->st->codec.height * 3) / 2;
                        ret = avcodec_decode_video(&ist->st->codec, 
                                                   &big_picture, &got_picture, ptr, len);
                        picture= *(AVPicture*)&big_picture;
                        ist->st->quality= big_picture.quality;
                        if (ret < 0) {
                        fail_decode:
                            fprintf(stderr, "Error while decoding stream #%d.%d\n",
                                    ist->file_index, ist->index);
                            av_free_packet(&pkt);
                            goto redo;
                        if (!got_picture) {
                            /* no picture yet */
                            ptr += ret;
                            len -= ret;
                            continue;
                    break;
                default:
                    goto fail_decode;
            } else {
                data_buf = ptr;
                data_size = len;
                ret = len;
            ptr += ret;
            len -= ret;
            buffer_to_free = 0;
            if (ist->st->codec.codec_type == CODEC_TYPE_VIDEO) {
                pre_process_video_frame(ist, &picture, &buffer_to_free);
            ist->frame_decoded = 1;
            /* frame rate emulation */
            if (ist->st->codec.rate_emu) {
                int64_t pts = av_rescale((int64_t) ist->frame * ist->st->codec.frame_rate_base, 1000000, ist->st->codec.frame_rate);
                int64_t now = av_gettime() - ist->start;
                if (pts > now)
                    usleep(pts - now);
                ist->frame++;
#if 0
            /* mpeg PTS deordering : if it is a P or I frame, the PTS
               is the one of the next displayed one */
            /* XXX: add mpeg4 too ? */
            if (ist->st->codec.codec_id == CODEC_ID_MPEG1VIDEO) {
                if (ist->st->codec.pict_type != B_TYPE) {
                    int64_t tmp;
                    tmp = ist->last_ip_pts;
                    ist->last_ip_pts  = ist->frac_pts.val;
                    ist->frac_pts.val = tmp;
#endif
            /* transcode raw format, encode packets and output them */
            for(i=0;i<nb_ostreams;i++) {
                int frame_size;
                ost = ost_table[i];
                if (ost->source_index == ist_index) {
                    os = output_files[ost->file_index];
                    if (ipts != AV_NOPTS_VALUE) {
#if 0
                        printf("%d: got pts=%f %f\n", 
                               i, pkt.pts / 90000.0, 
                               (ipts - ost->st->pts.val) / 90000.0);
#endif
                        /* set the input output pts pairs */
                        ost->sync_ipts = (double)ipts * is->pts_num / 
                            is->pts_den;
                        /* XXX: take into account the various fifos,
                           in particular for audio */
                        ost->sync_opts = ost->st->pts.val;
                        //printf("ipts=%lld sync_ipts=%f sync_opts=%lld pts.val=%lld pkt.pts=%lld\n", ipts, ost->sync_ipts, ost->sync_opts, ost->st->pts.val, pkt.pts); 
                    } else {
                        //printf("pts.val=%lld\n", ost->st->pts.val); 
                        ost->sync_ipts = AV_NOPTS_VALUE;
                    if (ost->encoding_needed) {
                        switch(ost->st->codec.codec_type) {
                        case CODEC_TYPE_AUDIO:
                            do_audio_out(os, ost, ist, data_buf, data_size);
                            break;
                        case CODEC_TYPE_VIDEO:
                            /* find an audio stream for synchro */
                            {
                                int i;
                                AVOutputStream *audio_sync, *ost1;
                                audio_sync = NULL;
                                for(i=0;i<nb_ostreams;i++) {
                                    ost1 = ost_table[i];
                                    if (ost1->file_index == ost->file_index &&
                                        ost1->st->codec.codec_type == CODEC_TYPE_AUDIO) {
                                        audio_sync = ost1;
                                        break;
                                do_video_out(os, ost, ist, &picture, &frame_size, audio_sync);
                                if (do_vstats && frame_size)
                                    do_video_stats(os, ost, frame_size);
                            break;
                        default:
                            av_abort();
                    } else {
                        AVFrame avframe;
                        /* no reencoding needed : output the packet directly */
                        /* force the input stream PTS */
                        memset(&avframe, 0, sizeof(AVFrame));
                        ost->st->codec.coded_frame= &avframe;
			avframe.key_frame = pkt.flags & PKT_FLAG_KEY; 
                        av_write_frame(os, ost->index, data_buf, data_size);
			ost->st->codec.frame_number++;
			ost->frame_number++;
            av_free(buffer_to_free);
            ipts = AV_NOPTS_VALUE;
    discard_packet:
        av_free_packet(&pkt);
        /* dump report by using the output first video and audio streams */
        print_report(output_files, ost_table, nb_ostreams, 0);
    term_exit();
    /* dump report by using the first video and audio streams */
    print_report(output_files, ost_table, nb_ostreams, 1);
    /* close each encoder */
    for(i=0;i<nb_ostreams;i++) {
        ost = ost_table[i];
        if (ost->encoding_needed) {
            av_freep(&ost->st->codec.stats_in);
            avcodec_close(&ost->st->codec);
    /* close each decoder */
    for(i=0;i<nb_istreams;i++) {
        ist = ist_table[i];
        if (ist->decoding_needed) {
            avcodec_close(&ist->st->codec);
    /* write the trailer if needed and close file */
    for(i=0;i<nb_output_files;i++) {
        os = output_files[i];
        av_write_trailer(os);
    /* finished ! */
    ret = 0;
 fail1:
    av_free(file_table);
    if (ist_table) {
        for(i=0;i<nb_istreams;i++) {
            ist = ist_table[i];
            av_free(ist);
        av_free(ist_table);
    if (ost_table) {
        for(i=0;i<nb_ostreams;i++) {
            ost = ost_table[i];
            if (ost) {
                if (ost->logfile) {
                    fclose(ost->logfile);
                    ost->logfile = NULL;
                fifo_free(&ost->fifo); /* works even if fifo is not
                                          initialized but set to zero */
                av_free(ost->pict_tmp.data[0]);
                if (ost->video_resample)
                    img_resample_close(ost->img_resample_ctx);
                if (ost->audio_resample)
                    audio_resample_close(ost->resample);
                av_free(ost);
        av_free(ost_table);
    return ret;
 fail:
    ret = -ENOMEM;
    goto fail1;