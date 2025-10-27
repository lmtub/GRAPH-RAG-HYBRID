static av_cold int omx_try_load(OMXContext *s, void *logctx,

                                const char *libname, const char *prefix)

{

    s->lib = dlopen(libname, RTLD_NOW | RTLD_GLOBAL);

    if (!s->lib) {

        av_log(logctx, AV_LOG_WARNING, "%s not found\n", libname);

        return AVERROR_ENCODER_NOT_FOUND;

    }

    s->ptr_Init                = dlsym_prefixed(s->lib, "OMX_Init", prefix);

    s->ptr_Deinit              = dlsym_prefixed(s->lib, "OMX_Deinit", prefix);

    s->ptr_ComponentNameEnum   = dlsym_prefixed(s->lib, "OMX_ComponentNameEnum", prefix);

    s->ptr_GetHandle           = dlsym_prefixed(s->lib, "OMX_GetHandle", prefix);

    s->ptr_FreeHandle          = dlsym_prefixed(s->lib, "OMX_FreeHandle", prefix);

    s->ptr_GetComponentsOfRole = dlsym_prefixed(s->lib, "OMX_GetComponentsOfRole", prefix);

    s->ptr_GetRolesOfComponent = dlsym_prefixed(s->lib, "OMX_GetRolesOfComponent", prefix);

    if (!s->ptr_Init || !s->ptr_Deinit || !s->ptr_ComponentNameEnum ||

        !s->ptr_GetHandle || !s->ptr_FreeHandle ||

        !s->ptr_GetComponentsOfRole || !s->ptr_GetRolesOfComponent) {

        av_log(logctx, AV_LOG_WARNING, "Not all functions found in %s\n", libname);

        dlclose(s->lib);

        s->lib = NULL;

        return AVERROR_ENCODER_NOT_FOUND;

    }

    return 0;

}
