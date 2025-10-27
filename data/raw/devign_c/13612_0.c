static void build_file_streams(void)

{

    FFServerStream *stream, *stream_next;

    int i, ret;



    /* gather all streams */

    for(stream = config.first_stream; stream; stream = stream_next) {

        AVFormatContext *infile = NULL;

        stream_next = stream->next;

        if (stream->stream_type == STREAM_TYPE_LIVE &&

            !stream->feed) {

            /* the stream comes from a file */

            /* try to open the file */

            /* open stream */

            if (stream->fmt && !strcmp(stream->fmt->name, "rtp")) {

                /* specific case : if transport stream output to RTP,

                   we use a raw transport stream reader */

                av_dict_set(&stream->in_opts, "mpeg2ts_compute_pcr", "1", 0);

            }



            if (!stream->feed_filename[0]) {

                http_log("Unspecified feed file for stream '%s'\n",

                         stream->filename);

                goto fail;

            }



            http_log("Opening feed file '%s' for stream '%s'\n",

                     stream->feed_filename, stream->filename);

            if ((ret = avformat_open_input(&infile, stream->feed_filename, stream->ifmt, &stream->in_opts)) < 0) {

                http_log("Could not open '%s': %s\n", stream->feed_filename,

                         av_err2str(ret));

                /* remove stream (no need to spend more time on it) */

            fail:

                remove_stream(stream);

            } else {

                /* find all the AVStreams inside and reference them in

                   'stream' */

                if (avformat_find_stream_info(infile, NULL) < 0) {

                    http_log("Could not find codec parameters from '%s'\n",

                             stream->feed_filename);

                    avformat_close_input(&infile);

                    goto fail;

                }

                extract_mpeg4_header(infile);



                for(i=0;i<infile->nb_streams;i++)

                    add_av_stream1(stream, infile->streams[i]->codec, 1);



                avformat_close_input(&infile);

            }

        }

    }

}
