static void build_feed_streams(void)

{

    FFStream *stream, *feed;

    int i;



    /* gather all streams */

    for(stream = first_stream; stream != NULL; stream = stream->next) {

        feed = stream->feed;

        if (feed) {

            if (!stream->is_feed) {

                /* we handle a stream coming from a feed */

                for(i=0;i<stream->nb_streams;i++)

                    stream->feed_streams[i] = add_av_stream(feed, stream->streams[i]);

            }

        }

    }



    /* gather all streams */

    for(stream = first_stream; stream != NULL; stream = stream->next) {

        feed = stream->feed;

        if (feed) {

            if (stream->is_feed) {

                for(i=0;i<stream->nb_streams;i++)

                    stream->feed_streams[i] = i;

            }

        }

    }



    /* create feed files if needed */

    for(feed = first_feed; feed != NULL; feed = feed->next_feed) {

        int fd;



        if (url_exist(feed->feed_filename)) {

            /* See if it matches */

            AVFormatContext *s;

            int matches = 0;



            if (av_open_input_file(&s, feed->feed_filename, NULL, FFM_PACKET_SIZE, NULL) >= 0) {

                /* Now see if it matches */

                if (s->nb_streams == feed->nb_streams) {

                    matches = 1;

                    for(i=0;i<s->nb_streams;i++) {

                        AVStream *sf, *ss;

                        sf = feed->streams[i];

                        ss = s->streams[i];



                        if (sf->index != ss->index ||

                            sf->id != ss->id) {

                            http_log("Index & Id do not match for stream %d (%s)\n",

                                   i, feed->feed_filename);

                            matches = 0;

                        } else {

                            AVCodecContext *ccf, *ccs;



                            ccf = sf->codec;

                            ccs = ss->codec;

#define CHECK_CODEC(x)  (ccf->x != ccs->x)



                            if (CHECK_CODEC(codec_id) || CHECK_CODEC(codec_type)) {

                                http_log("Codecs do not match for stream %d\n", i);

                                matches = 0;

                            } else if (CHECK_CODEC(bit_rate) || CHECK_CODEC(flags)) {

                                http_log("Codec bitrates do not match for stream %d\n", i);

                                matches = 0;

                            } else if (ccf->codec_type == AVMEDIA_TYPE_VIDEO) {

                                if (CHECK_CODEC(time_base.den) ||

                                    CHECK_CODEC(time_base.num) ||

                                    CHECK_CODEC(width) ||

                                    CHECK_CODEC(height)) {

                                    http_log("Codec width, height and framerate do not match for stream %d\n", i);

                                    matches = 0;

                                }

                            } else if (ccf->codec_type == AVMEDIA_TYPE_AUDIO) {

                                if (CHECK_CODEC(sample_rate) ||

                                    CHECK_CODEC(channels) ||

                                    CHECK_CODEC(frame_size)) {

                                    http_log("Codec sample_rate, channels, frame_size do not match for stream %d\n", i);

                                    matches = 0;

                                }

                            } else {

                                http_log("Unknown codec type\n");

                                matches = 0;

                            }

                        }

                        if (!matches)

                            break;

                    }

                } else

                    http_log("Deleting feed file '%s' as stream counts differ (%d != %d)\n",

                        feed->feed_filename, s->nb_streams, feed->nb_streams);



                av_close_input_file(s);

            } else

                http_log("Deleting feed file '%s' as it appears to be corrupt\n",

                        feed->feed_filename);



            if (!matches) {

                if (feed->readonly) {

                    http_log("Unable to delete feed file '%s' as it is marked readonly\n",

                        feed->feed_filename);

                    exit(1);

                }

                unlink(feed->feed_filename);

            }

        }

        if (!url_exist(feed->feed_filename)) {

            AVFormatContext s1 = {0}, *s = &s1;



            if (feed->readonly) {

                http_log("Unable to create feed file '%s' as it is marked readonly\n",

                    feed->feed_filename);

                exit(1);

            }



            /* only write the header of the ffm file */

            if (avio_open(&s->pb, feed->feed_filename, AVIO_FLAG_WRITE) < 0) {

                http_log("Could not open output feed file '%s'\n",

                         feed->feed_filename);

                exit(1);

            }

            s->oformat = feed->fmt;

            s->nb_streams = feed->nb_streams;

            for(i=0;i<s->nb_streams;i++) {

                AVStream *st;

                st = feed->streams[i];

                s->streams[i] = st;

            }

            av_set_parameters(s, NULL);

            if (av_write_header(s) < 0) {

                http_log("Container doesn't supports the required parameters\n");

                exit(1);

            }

            /* XXX: need better api */

            av_freep(&s->priv_data);

            avio_close(s->pb);

        }

        /* get feed size and write index */

        fd = open(feed->feed_filename, O_RDONLY);

        if (fd < 0) {

            http_log("Could not open output feed file '%s'\n",

                    feed->feed_filename);

            exit(1);

        }



        feed->feed_write_index = FFMAX(ffm_read_write_index(fd), FFM_PACKET_SIZE);

        feed->feed_size = lseek(fd, 0, SEEK_END);

        /* ensure that we do not wrap before the end of file */

        if (feed->feed_max_size && feed->feed_max_size < feed->feed_size)

            feed->feed_max_size = feed->feed_size;



        close(fd);

    }

}
