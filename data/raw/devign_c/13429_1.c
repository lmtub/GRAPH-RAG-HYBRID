static int dshow_read_header(AVFormatContext *avctx)

{

    struct dshow_ctx *ctx = avctx->priv_data;

    IGraphBuilder *graph = NULL;

    ICreateDevEnum *devenum = NULL;

    IMediaControl *control = NULL;

    IMediaEvent *media_event = NULL;

    HANDLE media_event_handle;

    HANDLE proc;

    int ret = AVERROR(EIO);

    int r;



    CoInitialize(0);



    if (!ctx->list_devices && !parse_device_name(avctx)) {

        av_log(avctx, AV_LOG_ERROR, "Malformed dshow input string.\n");

        goto error;

    }



    ctx->video_codec_id = avctx->video_codec_id ? avctx->video_codec_id

                                                : AV_CODEC_ID_RAWVIDEO;

    if (ctx->pixel_format != AV_PIX_FMT_NONE) {

        if (ctx->video_codec_id != AV_CODEC_ID_RAWVIDEO) {

            av_log(avctx, AV_LOG_ERROR, "Pixel format may only be set when "

                              "video codec is not set or set to rawvideo\n");

            ret = AVERROR(EINVAL);

            goto error;

        }

    }

    if (ctx->framerate) {

        r = av_parse_video_rate(&ctx->requested_framerate, ctx->framerate);

        if (r < 0) {

            av_log(avctx, AV_LOG_ERROR, "Could not parse framerate '%s'.\n", ctx->framerate);

            goto error;

        }

    }



    r = CoCreateInstance(&CLSID_FilterGraph, NULL, CLSCTX_INPROC_SERVER,

                         &IID_IGraphBuilder, (void **) &graph);

    if (r != S_OK) {

        av_log(avctx, AV_LOG_ERROR, "Could not create capture graph.\n");

        goto error;

    }

    ctx->graph = graph;



    r = CoCreateInstance(&CLSID_SystemDeviceEnum, NULL, CLSCTX_INPROC_SERVER,

                         &IID_ICreateDevEnum, (void **) &devenum);

    if (r != S_OK) {

        av_log(avctx, AV_LOG_ERROR, "Could not enumerate system devices.\n");

        goto error;

    }



    if (ctx->list_devices) {

        av_log(avctx, AV_LOG_INFO, "DirectShow video devices\n");

        dshow_cycle_devices(avctx, devenum, VideoDevice, NULL);

        av_log(avctx, AV_LOG_INFO, "DirectShow audio devices\n");

        dshow_cycle_devices(avctx, devenum, AudioDevice, NULL);

        ret = AVERROR_EXIT;

        goto error;

    }

    if (ctx->list_options) {

        if (ctx->device_name[VideoDevice])

            dshow_list_device_options(avctx, devenum, VideoDevice);

        if (ctx->device_name[AudioDevice])

            dshow_list_device_options(avctx, devenum, AudioDevice);

        ret = AVERROR_EXIT;

        goto error;

    }



    if (ctx->device_name[VideoDevice]) {

        if ((r = dshow_open_device(avctx, devenum, VideoDevice)) < 0 ||

            (r = dshow_add_device(avctx, VideoDevice)) < 0) {

            ret = r;

            goto error;

        }

    }

    if (ctx->device_name[AudioDevice]) {

        if ((r = dshow_open_device(avctx, devenum, AudioDevice)) < 0 ||

            (r = dshow_add_device(avctx, AudioDevice)) < 0) {

            ret = r;

            goto error;

        }

    }



    ctx->mutex = CreateMutex(NULL, 0, NULL);

    if (!ctx->mutex) {

        av_log(avctx, AV_LOG_ERROR, "Could not create Mutex\n");

        goto error;

    }

    ctx->event[1] = CreateEvent(NULL, 1, 0, NULL);

    if (!ctx->event[1]) {

        av_log(avctx, AV_LOG_ERROR, "Could not create Event\n");

        goto error;

    }



    r = IGraphBuilder_QueryInterface(graph, &IID_IMediaControl, (void **) &control);

    if (r != S_OK) {

        av_log(avctx, AV_LOG_ERROR, "Could not get media control.\n");

        goto error;

    }

    ctx->control = control;



    r = IGraphBuilder_QueryInterface(graph, &IID_IMediaEvent, (void **) &media_event);

    if (r != S_OK) {

        av_log(avctx, AV_LOG_ERROR, "Could not get media event.\n");

        goto error;

    }

    ctx->media_event = media_event;



    r = IMediaEvent_GetEventHandle(media_event, (void *) &media_event_handle);

    if (r != S_OK) {

        av_log(avctx, AV_LOG_ERROR, "Could not get media event handle.\n");

        goto error;

    }

    proc = GetCurrentProcess();

    r = DuplicateHandle(proc, media_event_handle, proc, &ctx->event[0],

                        0, 0, DUPLICATE_SAME_ACCESS);

    if (!r) {

        av_log(avctx, AV_LOG_ERROR, "Could not duplicate media event handle.\n");

        goto error;

    }



    r = IMediaControl_Run(control);

    if (r == S_FALSE) {

        OAFilterState pfs;

        r = IMediaControl_GetState(control, 0, &pfs);

    }

    if (r != S_OK) {

        av_log(avctx, AV_LOG_ERROR, "Could not run filter\n");

        goto error;

    }



    ret = 0;



error:



    if (devenum)

        ICreateDevEnum_Release(devenum);



    if (ret < 0)

        dshow_read_close(avctx);



    return ret;

}
