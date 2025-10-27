dshow_cycle_devices(AVFormatContext *avctx, ICreateDevEnum *devenum,

                    enum dshowDeviceType devtype, IBaseFilter **pfilter)

{

    struct dshow_ctx *ctx = avctx->priv_data;

    IBaseFilter *device_filter = NULL;

    IEnumMoniker *classenum = NULL;

    IMoniker *m = NULL;

    const char *device_name = ctx->device_name[devtype];

    int skip = (devtype == VideoDevice) ? ctx->video_device_number

                                        : ctx->audio_device_number;

    int r;



    const GUID *device_guid[2] = { &CLSID_VideoInputDeviceCategory,

                                   &CLSID_AudioInputDeviceCategory };

    const char *devtypename = (devtype == VideoDevice) ? "video" : "audio";



    r = ICreateDevEnum_CreateClassEnumerator(devenum, device_guid[devtype],

                                             (IEnumMoniker **) &classenum, 0);

    if (r != S_OK) {

        av_log(avctx, AV_LOG_ERROR, "Could not enumerate %s devices.\n",

               devtypename);

        return AVERROR(EIO);

    }



    while (!device_filter && IEnumMoniker_Next(classenum, 1, &m, NULL) == S_OK) {

        IPropertyBag *bag = NULL;

        char *buf = NULL;

        VARIANT var;



        r = IMoniker_BindToStorage(m, 0, 0, &IID_IPropertyBag, (void *) &bag);

        if (r != S_OK)

            goto fail1;



        var.vt = VT_BSTR;

        r = IPropertyBag_Read(bag, L"FriendlyName", &var, NULL);

        if (r != S_OK)

            goto fail1;



        buf = dup_wchar_to_utf8(var.bstrVal);



        if (pfilter) {

            if (strcmp(device_name, buf))

                goto fail1;



            if (!skip--)

                IMoniker_BindToObject(m, 0, 0, &IID_IBaseFilter, (void *) &device_filter);

        } else {

            av_log(avctx, AV_LOG_INFO, " \"%s\"\n", buf);

        }



fail1:

        if (buf)

            av_free(buf);

        if (bag)

            IPropertyBag_Release(bag);

        IMoniker_Release(m);

    }



    IEnumMoniker_Release(classenum);



    if (pfilter) {

        if (!device_filter) {

            av_log(avctx, AV_LOG_ERROR, "Could not find %s device.\n",

                   devtypename);

            return AVERROR(EIO);

        }

        *pfilter = device_filter;

    }



    return 0;

}
