int av_open_input_stream(AVFormatContext **ic_ptr,

                         ByteIOContext *pb, const char *filename,

                         AVInputFormat *fmt, AVFormatParameters *ap)

{

    int err;

    AVFormatContext *ic;

    AVFormatParameters default_ap;



    if(!ap){

        ap=&default_ap;

        memset(ap, 0, sizeof(default_ap));

    }



    if(!ap->prealloced_context)

        ic = avformat_alloc_context();

    else

        ic = *ic_ptr;

    if (!ic) {

        err = AVERROR(ENOMEM);

        goto fail;

    }

    ic->iformat = fmt;

    ic->pb = pb;

    ic->duration = AV_NOPTS_VALUE;

    ic->start_time = AV_NOPTS_VALUE;

    av_strlcpy(ic->filename, filename, sizeof(ic->filename));



    /* allocate private data */

    if (fmt->priv_data_size > 0) {

        ic->priv_data = av_mallocz(fmt->priv_data_size);

        if (!ic->priv_data) {

            err = AVERROR(ENOMEM);

            goto fail;

        }

    } else {

        ic->priv_data = NULL;

    }



    if (ic->iformat->read_header) {

        err = ic->iformat->read_header(ic, ap);

        if (err < 0)

            goto fail;

    }



    if (pb && !ic->data_offset)

        ic->data_offset = url_ftell(ic->pb);



#if LIBAVFORMAT_VERSION_MAJOR < 53

    ff_metadata_demux_compat(ic);

#endif



    ic->raw_packet_buffer_remaining_size = RAW_PACKET_BUFFER_SIZE;



    *ic_ptr = ic;

    return 0;

 fail:

    if (ic) {

        int i;

        av_freep(&ic->priv_data);

        for(i=0;i<ic->nb_streams;i++) {

            AVStream *st = ic->streams[i];

            if (st) {

                av_free(st->priv_data);

                av_free(st->codec->extradata);


            }

            av_free(st);

        }

    }

    av_free(ic);

    *ic_ptr = NULL;

    return err;

}