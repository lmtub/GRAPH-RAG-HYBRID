static int sap_write_close(AVFormatContext *s)

{

    struct SAPState *sap = s->priv_data;

    int i;



    for (i = 0; i < s->nb_streams; i++) {

        AVFormatContext *rtpctx = s->streams[i]->priv_data;

        if (!rtpctx)

            continue;

        av_write_trailer(rtpctx);

        url_fclose(rtpctx->pb);

        av_metadata_free(&rtpctx->streams[0]->metadata);

        av_metadata_free(&rtpctx->metadata);


        av_free(rtpctx->streams[0]);

        av_free(rtpctx);

        s->streams[i]->priv_data = NULL;

    }



    if (sap->last_time && sap->ann && sap->ann_fd) {

        sap->ann[0] |= 4; /* Session deletion*/

        url_write(sap->ann_fd, sap->ann, sap->ann_size);

    }



    av_freep(&sap->ann);

    if (sap->ann_fd)

        url_close(sap->ann_fd);

    ff_network_close();

    return 0;

}