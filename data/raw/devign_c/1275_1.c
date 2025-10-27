static void compute_stats(HTTPContext *c)

{

    HTTPContext *c1;

    FFStream *stream;

    char *p;

    time_t ti;

    int i, len;

    ByteIOContext pb1, *pb = &pb1;



    if (url_open_dyn_buf(pb) < 0) {

        /* XXX: return an error ? */

        c->buffer_ptr = c->buffer;

        c->buffer_end = c->buffer;

        return;

    }



    url_fprintf(pb, "HTTP/1.0 200 OK\r\n");

    url_fprintf(pb, "Content-type: %s\r\n", "text/html");

    url_fprintf(pb, "Pragma: no-cache\r\n");

    url_fprintf(pb, "\r\n");



    url_fprintf(pb, "<HEAD><TITLE>FFServer Status</TITLE>\n");

    if (c->stream->feed_filename)

        url_fprintf(pb, "<link rel=\"shortcut icon\" href=\"%s\">\n", c->stream->feed_filename);

    url_fprintf(pb, "</HEAD>\n<BODY>");

    url_fprintf(pb, "<H1>FFServer Status</H1>\n");

    /* format status */

    url_fprintf(pb, "<H2>Available Streams</H2>\n");

    url_fprintf(pb, "<TABLE cellspacing=0 cellpadding=4>\n");

    url_fprintf(pb, "<TR><Th valign=top>Path<th align=left>Served<br>Conns<Th><br>bytes<Th valign=top>Format<Th>Bit rate<br>kbits/s<Th align=left>Video<br>kbits/s<th><br>Codec<Th align=left>Audio<br>kbits/s<th><br>Codec<Th align=left valign=top>Feed\n");

    stream = first_stream;

    while (stream != NULL) {

        char sfilename[1024];

        char *eosf;



        if (stream->feed != stream) {

            av_strlcpy(sfilename, stream->filename, sizeof(sfilename) - 10);

            eosf = sfilename + strlen(sfilename);

            if (eosf - sfilename >= 4) {

                if (strcmp(eosf - 4, ".asf") == 0)

                    strcpy(eosf - 4, ".asx");

                else if (strcmp(eosf - 3, ".rm") == 0)

                    strcpy(eosf - 3, ".ram");

                else if (!strcmp(stream->fmt->name, "rtp")) {

                    /* generate a sample RTSP director if

                       unicast. Generate an SDP redirector if

                       multicast */

                    eosf = strrchr(sfilename, '.');

                    if (!eosf)

                        eosf = sfilename + strlen(sfilename);

                    if (stream->is_multicast)

                        strcpy(eosf, ".sdp");

                    else

                        strcpy(eosf, ".rtsp");

                }

            }



            url_fprintf(pb, "<TR><TD><A HREF=\"/%s\">%s</A> ",

                         sfilename, stream->filename);

            url_fprintf(pb, "<td align=right> %d <td align=right> ",

                        stream->conns_served);

            fmt_bytecount(pb, stream->bytes_served);

            switch(stream->stream_type) {

            case STREAM_TYPE_LIVE:

                {

                    int audio_bit_rate = 0;

                    int video_bit_rate = 0;

                    const char *audio_codec_name = "";

                    const char *video_codec_name = "";

                    const char *audio_codec_name_extra = "";

                    const char *video_codec_name_extra = "";



                    for(i=0;i<stream->nb_streams;i++) {

                        AVStream *st = stream->streams[i];

                        AVCodec *codec = avcodec_find_encoder(st->codec->codec_id);

                        switch(st->codec->codec_type) {

                        case CODEC_TYPE_AUDIO:

                            audio_bit_rate += st->codec->bit_rate;

                            if (codec) {

                                if (*audio_codec_name)

                                    audio_codec_name_extra = "...";

                                audio_codec_name = codec->name;

                            }

                            break;

                        case CODEC_TYPE_VIDEO:

                            video_bit_rate += st->codec->bit_rate;

                            if (codec) {

                                if (*video_codec_name)

                                    video_codec_name_extra = "...";

                                video_codec_name = codec->name;

                            }

                            break;

                        case CODEC_TYPE_DATA:

                            video_bit_rate += st->codec->bit_rate;

                            break;

                        default:

                            abort();

                        }

                    }

                    url_fprintf(pb, "<TD align=center> %s <TD align=right> %d <TD align=right> %d <TD> %s %s <TD align=right> %d <TD> %s %s",

                                 stream->fmt->name,

                                 stream->bandwidth,

                                 video_bit_rate / 1000, video_codec_name, video_codec_name_extra,

                                 audio_bit_rate / 1000, audio_codec_name, audio_codec_name_extra);

                    if (stream->feed)

                        url_fprintf(pb, "<TD>%s", stream->feed->filename);

                    else

                        url_fprintf(pb, "<TD>%s", stream->feed_filename);

                    url_fprintf(pb, "\n");

                }

                break;

            default:

                url_fprintf(pb, "<TD align=center> - <TD align=right> - <TD align=right> - <td><td align=right> - <TD>\n");

                break;

            }

        }

        stream = stream->next;

    }

    url_fprintf(pb, "</TABLE>\n");



    stream = first_stream;

    while (stream != NULL) {

        if (stream->feed == stream) {

            url_fprintf(pb, "<h2>Feed %s</h2>", stream->filename);

            if (stream->pid) {

                url_fprintf(pb, "Running as pid %d.\n", stream->pid);



#if defined(linux) && !defined(CONFIG_NOCUTILS)

                {

                    FILE *pid_stat;

                    char ps_cmd[64];



                    /* This is somewhat linux specific I guess */

                    snprintf(ps_cmd, sizeof(ps_cmd),

                             "ps -o \"%%cpu,cputime\" --no-headers %d",

                             stream->pid);



                    pid_stat = popen(ps_cmd, "r");

                    if (pid_stat) {

                        char cpuperc[10];

                        char cpuused[64];



                        if (fscanf(pid_stat, "%10s %64s", cpuperc,

                                   cpuused) == 2) {

                            url_fprintf(pb, "Currently using %s%% of the cpu. Total time used %s.\n",

                                         cpuperc, cpuused);

                        }

                        fclose(pid_stat);

                    }

                }

#endif



                url_fprintf(pb, "<p>");

            }

            url_fprintf(pb, "<table cellspacing=0 cellpadding=4><tr><th>Stream<th>type<th>kbits/s<th align=left>codec<th align=left>Parameters\n");



            for (i = 0; i < stream->nb_streams; i++) {

                AVStream *st = stream->streams[i];

                AVCodec *codec = avcodec_find_encoder(st->codec->codec_id);

                const char *type = "unknown";

                char parameters[64];



                parameters[0] = 0;



                switch(st->codec->codec_type) {

                case CODEC_TYPE_AUDIO:

                    type = "audio";

                    snprintf(parameters, sizeof(parameters), "%d channel(s), %d Hz", st->codec->channels, st->codec->sample_rate);

                    break;

                case CODEC_TYPE_VIDEO:

                    type = "video";

                    snprintf(parameters, sizeof(parameters), "%dx%d, q=%d-%d, fps=%d", st->codec->width, st->codec->height,

                                st->codec->qmin, st->codec->qmax, st->codec->time_base.den / st->codec->time_base.num);

                    break;

                default:

                    abort();

                }

                url_fprintf(pb, "<tr><td align=right>%d<td>%s<td align=right>%d<td>%s<td>%s\n",

                        i, type, st->codec->bit_rate/1000, codec ? codec->name : "", parameters);

            }

            url_fprintf(pb, "</table>\n");



        }

        stream = stream->next;

    }



#if 0

    {

        float avg;

        AVCodecContext *enc;

        char buf[1024];



        /* feed status */

        stream = first_feed;

        while (stream != NULL) {

            url_fprintf(pb, "<H1>Feed '%s'</H1>\n", stream->filename);

            url_fprintf(pb, "<TABLE>\n");

            url_fprintf(pb, "<TR><TD>Parameters<TD>Frame count<TD>Size<TD>Avg bitrate (kbits/s)\n");

            for(i=0;i<stream->nb_streams;i++) {

                AVStream *st = stream->streams[i];

                FeedData *fdata = st->priv_data;

                enc = st->codec;



                avcodec_string(buf, sizeof(buf), enc);

                avg = fdata->avg_frame_size * (float)enc->rate * 8.0;

                if (enc->codec->type == CODEC_TYPE_AUDIO && enc->frame_size > 0)

                    avg /= enc->frame_size;

                url_fprintf(pb, "<TR><TD>%s <TD> %d <TD> %"PRId64" <TD> %0.1f\n",

                             buf, enc->frame_number, fdata->data_count, avg / 1000.0);

            }

            url_fprintf(pb, "</TABLE>\n");

            stream = stream->next_feed;

        }

    }

#endif



    /* connection status */

    url_fprintf(pb, "<H2>Connection Status</H2>\n");



    url_fprintf(pb, "Number of connections: %d / %d<BR>\n",

                 nb_connections, nb_max_connections);



    url_fprintf(pb, "Bandwidth in use: %dk / %dk<BR>\n",

                 current_bandwidth, max_bandwidth);



    url_fprintf(pb, "<TABLE>\n");

    url_fprintf(pb, "<TR><th>#<th>File<th>IP<th>Proto<th>State<th>Target bits/sec<th>Actual bits/sec<th>Bytes transferred\n");

    c1 = first_http_ctx;

    i = 0;

    while (c1 != NULL) {

        int bitrate;

        int j;



        bitrate = 0;

        if (c1->stream) {

            for (j = 0; j < c1->stream->nb_streams; j++) {

                if (!c1->stream->feed)

                    bitrate += c1->stream->streams[j]->codec->bit_rate;

                else if (c1->feed_streams[j] >= 0)

                    bitrate += c1->stream->feed->streams[c1->feed_streams[j]]->codec->bit_rate;

            }

        }



        i++;

        p = inet_ntoa(c1->from_addr.sin_addr);

        url_fprintf(pb, "<TR><TD><B>%d</B><TD>%s%s<TD>%s<TD>%s<TD>%s<td align=right>",

                    i,

                    c1->stream ? c1->stream->filename : "",

                    c1->state == HTTPSTATE_RECEIVE_DATA ? "(input)" : "",

                    p,

                    c1->protocol,

                    http_state[c1->state]);

        fmt_bytecount(pb, bitrate);

        url_fprintf(pb, "<td align=right>");

        fmt_bytecount(pb, compute_datarate(&c1->datarate, c1->data_count) * 8);

        url_fprintf(pb, "<td align=right>");

        fmt_bytecount(pb, c1->data_count);

        url_fprintf(pb, "\n");

        c1 = c1->next;

    }

    url_fprintf(pb, "</TABLE>\n");



    /* date */

    ti = time(NULL);

    p = ctime(&ti);

    url_fprintf(pb, "<HR size=1 noshade>Generated at %s", p);

    url_fprintf(pb, "</BODY>\n</HTML>\n");



    len = url_close_dyn_buf(pb, &c->pb_buffer);

    c->buffer_ptr = c->pb_buffer;

    c->buffer_end = c->pb_buffer + len;

}
