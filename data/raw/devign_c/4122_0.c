static av_cold int find_component(OMXContext *omx_context, void *logctx,

                                  const char *role, char *str, int str_size)

{

    OMX_U32 i, num = 0;

    char **components;

    int ret = 0;



#if CONFIG_OMX_RPI

    if (av_strstart(role, "video_encoder.", NULL)) {

        av_strlcpy(str, "OMX.broadcom.video_encode", str_size);

        return 0;

    }

#endif

    omx_context->ptr_GetComponentsOfRole((OMX_STRING) role, &num, NULL);

    if (!num) {

        av_log(logctx, AV_LOG_WARNING, "No component for role %s found\n", role);

        return AVERROR_ENCODER_NOT_FOUND;

    }

    components = av_mallocz(sizeof(char*) * num);

    if (!components)

        return AVERROR(ENOMEM);

    for (i = 0; i < num; i++) {

        components[i] = av_mallocz(OMX_MAX_STRINGNAME_SIZE);

        if (!components) {

            ret = AVERROR(ENOMEM);

            goto end;

        }

    }

    omx_context->ptr_GetComponentsOfRole((OMX_STRING) role, &num, (OMX_U8**) components);

    av_strlcpy(str, components[0], str_size);

end:

    for (i = 0; i < num; i++)

        av_free(components[i]);

    av_free(components);

    return ret;

}
