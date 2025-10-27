static av_cold OMXContext *omx_init(void *logctx, const char *libname, const char *prefix)

{

    static const char * const libnames[] = {

        "libOMX_Core.so",

        "libOmxCore.so",

        NULL

    };

    const char* const* nameptr;

    int ret = AVERROR_ENCODER_NOT_FOUND;

    OMXContext *omx_context;



    omx_context = av_mallocz(sizeof(*omx_context));

    if (!omx_context)

        return NULL;

    if (libname) {

        ret = omx_try_load(omx_context, logctx, libname, prefix);

        if (ret < 0) {

            av_free(omx_context);

            return NULL;

        }

    } else {

        for (nameptr = libnames; *nameptr; nameptr++)

            if (!(ret = omx_try_load(omx_context, logctx, *nameptr, prefix)))

                break;

        if (!*nameptr) {

            av_free(omx_context);

            return NULL;

        }

    }



    omx_context->ptr_Init();

    return omx_context;

}
