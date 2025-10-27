static int dc1394_read_header(AVFormatContext *c, AVFormatParameters * ap)

{

    dc1394_data* dc1394 = c->priv_data;

    AVStream* vst;

    nodeid_t* camera_nodes;

    int res;

    struct dc1394_frame_format *fmt;

    struct dc1394_frame_rate *fps;



    for (fmt = dc1394_frame_formats; fmt->width; fmt++)

         if (fmt->pix_fmt == ap->pix_fmt && fmt->width == ap->width && fmt->height == ap->height)

             break;



    for (fps = dc1394_frame_rates; fps->frame_rate; fps++)

         if (fps->frame_rate == av_rescale(1000, ap->time_base.den, ap->time_base.num))

             break;



    /* create a video stream */

    vst = av_new_stream(c, 0);

    if (!vst)

        return -1;

    av_set_pts_info(vst, 64, 1, 1000);

    vst->codec->codec_type = CODEC_TYPE_VIDEO;

    vst->codec->codec_id = CODEC_ID_RAWVIDEO;

    vst->codec->time_base.den = fps->frame_rate;

    vst->codec->time_base.num = 1000;

    vst->codec->width = fmt->width;

    vst->codec->height = fmt->height;

    vst->codec->pix_fmt = fmt->pix_fmt;



    /* packet init */

    av_init_packet(&dc1394->packet);

    dc1394->packet.size = avpicture_get_size(fmt->pix_fmt, fmt->width, fmt->height);

    dc1394->packet.stream_index = vst->index;

    dc1394->packet.flags |= PKT_FLAG_KEY;



    dc1394->current_frame = 0;

    dc1394->fps = fps->frame_rate;



    vst->codec->bit_rate = av_rescale(dc1394->packet.size * 8, fps->frame_rate, 1000);



    /* Now lets prep the hardware */

    dc1394->handle = dc1394_create_handle(0); /* FIXME: gotta have ap->port */

    if (!dc1394->handle) {

        av_log(c, AV_LOG_ERROR, "Can't acquire dc1394 handle on port %d\n", 0 /* ap->port */);

        goto out;

    }

    camera_nodes = dc1394_get_camera_nodes(dc1394->handle, &res, 1);

    if (!camera_nodes || camera_nodes[ap->channel] == DC1394_NO_CAMERA) {

        av_log(c, AV_LOG_ERROR, "There's no IIDC camera on the channel %d\n", ap->channel);

        goto out_handle;

    }

    res = dc1394_dma_setup_capture(dc1394->handle, camera_nodes[ap->channel],

                                   0,

                                   FORMAT_VGA_NONCOMPRESSED,

                                   fmt->frame_size_id,

                                   SPEED_400,

                                   fps->frame_rate_id, 8, 1,

                                   c->filename,

                                   &dc1394->camera);

    dc1394_free_camera_nodes(camera_nodes);

    if (res != DC1394_SUCCESS) {

        av_log(c, AV_LOG_ERROR, "Can't prepare camera for the DMA capture\n");

        goto out_handle;

    }



    res = dc1394_start_iso_transmission(dc1394->handle, dc1394->camera.node);

    if (res != DC1394_SUCCESS) {

        av_log(c, AV_LOG_ERROR, "Can't start isochronous transmission\n");

        goto out_handle_dma;

    }



    return 0;



out_handle_dma:

    dc1394_dma_unlisten(dc1394->handle, &dc1394->camera);

    dc1394_dma_release_camera(dc1394->handle, &dc1394->camera);

out_handle:

    dc1394_destroy_handle(dc1394->handle);

out:

    return -1;

}
