static int vdpau_alloc(AVCodecContext *s)

{

    InputStream  *ist = s->opaque;

    int loglevel = (ist->hwaccel_id == HWACCEL_AUTO) ? AV_LOG_VERBOSE : AV_LOG_ERROR;

    AVVDPAUContext *vdpau_ctx;

    VDPAUContext *ctx;

    const char *display, *vendor;

    VdpStatus err;

    int i;



    ctx = av_mallocz(sizeof(*ctx));

    if (!ctx)

        return AVERROR(ENOMEM);



    ist->hwaccel_ctx           = ctx;

    ist->hwaccel_uninit        = vdpau_uninit;

    ist->hwaccel_get_buffer    = vdpau_get_buffer;

    ist->hwaccel_retrieve_data = vdpau_retrieve_data;



    ctx->tmp_frame = av_frame_alloc();

    if (!ctx->tmp_frame)

        goto fail;



    ctx->dpy = XOpenDisplay(ist->hwaccel_device);

    if (!ctx->dpy) {

        av_log(NULL, loglevel, "Cannot open the X11 display %s.\n",

               XDisplayName(ist->hwaccel_device));

        goto fail;

    }

    display = XDisplayString(ctx->dpy);



    err = vdp_device_create_x11(ctx->dpy, XDefaultScreen(ctx->dpy), &ctx->device,

                                &ctx->get_proc_address);

    if (err != VDP_STATUS_OK) {

        av_log(NULL, loglevel, "VDPAU device creation on X11 display %s failed.\n",

               display);

        goto fail;

    }



#define GET_CALLBACK(id, result)                                                \

do {                                                                            \

    void *tmp;                                                                  \

    err = ctx->get_proc_address(ctx->device, id, &tmp);                         \

    if (err != VDP_STATUS_OK) {                                                 \

        av_log(NULL, loglevel, "Error getting the " #id " callback.\n");        \

        goto fail;                                                              \

    }                                                                           \

    ctx->result = tmp;                                                          \

} while (0)



    GET_CALLBACK(VDP_FUNC_ID_GET_ERROR_STRING,               get_error_string);

    GET_CALLBACK(VDP_FUNC_ID_GET_INFORMATION_STRING,         get_information_string);

    GET_CALLBACK(VDP_FUNC_ID_DEVICE_DESTROY,                 device_destroy);

    if (vdpau_api_ver == 1) {

        GET_CALLBACK(VDP_FUNC_ID_DECODER_CREATE,                 decoder_create);

        GET_CALLBACK(VDP_FUNC_ID_DECODER_DESTROY,                decoder_destroy);

        GET_CALLBACK(VDP_FUNC_ID_DECODER_RENDER,                 decoder_render);

    }

    GET_CALLBACK(VDP_FUNC_ID_VIDEO_SURFACE_CREATE,           video_surface_create);

    GET_CALLBACK(VDP_FUNC_ID_VIDEO_SURFACE_DESTROY,          video_surface_destroy);

    GET_CALLBACK(VDP_FUNC_ID_VIDEO_SURFACE_GET_BITS_Y_CB_CR, video_surface_get_bits);

    GET_CALLBACK(VDP_FUNC_ID_VIDEO_SURFACE_GET_PARAMETERS,   video_surface_get_parameters);

    GET_CALLBACK(VDP_FUNC_ID_VIDEO_SURFACE_QUERY_GET_PUT_BITS_Y_CB_CR_CAPABILITIES,

                 video_surface_query);



    for (i = 0; i < FF_ARRAY_ELEMS(vdpau_formats); i++) {

        VdpBool supported;

        err = ctx->video_surface_query(ctx->device, VDP_CHROMA_TYPE_420,

                                       vdpau_formats[i][0], &supported);

        if (err != VDP_STATUS_OK) {

            av_log(NULL, loglevel,

                   "Error querying VDPAU surface capabilities: %s\n",

                   ctx->get_error_string(err));

            goto fail;

        }

        if (supported)

            break;

    }

    if (i == FF_ARRAY_ELEMS(vdpau_formats)) {

        av_log(NULL, loglevel,

               "No supported VDPAU format for retrieving the data.\n");

        return AVERROR(EINVAL);

    }

    ctx->vdpau_format = vdpau_formats[i][0];

    ctx->pix_fmt      = vdpau_formats[i][1];



    if (vdpau_api_ver == 1) {

        vdpau_ctx = av_vdpau_alloc_context();

        if (!vdpau_ctx)

            goto fail;

        vdpau_ctx->render = ctx->decoder_render;



        s->hwaccel_context = vdpau_ctx;

    } else

    if (av_vdpau_bind_context(s, ctx->device, ctx->get_proc_address, 0))

        goto fail;



    ctx->get_information_string(&vendor);

    av_log(NULL, AV_LOG_VERBOSE, "Using VDPAU -- %s -- on X11 display %s, "

           "to decode input stream #%d:%d.\n", vendor,

           display, ist->file_index, ist->st->index);



    return 0;



fail:

    av_log(NULL, loglevel, "VDPAU init failed for stream #%d:%d.\n",

           ist->file_index, ist->st->index);

    vdpau_uninit(s);

    return AVERROR(EINVAL);

}
