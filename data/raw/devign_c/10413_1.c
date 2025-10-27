static int print_device_sinks(AVOutputFormat *fmt, AVDictionary *opts)

{

    int ret, i;

    AVFormatContext *dev = NULL;

    AVDeviceInfoList *device_list = NULL;

    AVDictionary *tmp_opts = NULL;



    if (!fmt || !fmt->priv_class  || !AV_IS_OUTPUT_DEVICE(fmt->priv_class->category))

        return AVERROR(EINVAL);



    printf("Audo-detected sinks for %s:\n", fmt->name);

    if (!fmt->get_device_list) {

        ret = AVERROR(ENOSYS);

        printf("Cannot list sinks. Not implemented.\n");

        goto fail;

    }



    if ((ret = avformat_alloc_output_context2(&dev, fmt, NULL, NULL)) < 0) {

        printf("Cannot open device: %s.\n", fmt->name);

        goto fail;

    }

    av_dict_copy(&tmp_opts, opts, 0);

    av_opt_set_dict2(dev, &tmp_opts, AV_OPT_SEARCH_CHILDREN);



    if ((ret = avdevice_list_devices(dev, &device_list)) < 0) {

        printf("Cannot list sinks.\n");

        goto fail;

    }



    for (i = 0; i < device_list->nb_devices; i++) {

        printf("%s %s [%s]\n", device_list->default_device == i ? "*" : " ",

               device_list->devices[i]->device_name, device_list->devices[i]->device_description);

    }



  fail:

    av_dict_free(&tmp_opts);

    avdevice_free_list_devices(&device_list);

    avformat_free_context(dev);

    return ret;

}
