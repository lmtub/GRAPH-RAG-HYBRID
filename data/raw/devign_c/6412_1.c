static MpegTSService *mpegts_add_service(MpegTSWrite *ts, int sid,

                                         const char *provider_name,

                                         const char *name)

{

    MpegTSService *service;



    service = av_mallocz(sizeof(MpegTSService));

    if (!service)

        return NULL;

    service->pmt.pid       = ts->pmt_start_pid + ts->nb_services;

    service->sid           = sid;

    service->provider_name = av_strdup(provider_name);

    service->name          = av_strdup(name);

    service->pcr_pid       = 0x1fff;

    dynarray_add(&ts->services, &ts->nb_services, service);

    return service;

}
