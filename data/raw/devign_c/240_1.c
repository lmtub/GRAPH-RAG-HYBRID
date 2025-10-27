static VncServerInfo *vnc_server_info_get(VncDisplay *vd)

{

    VncServerInfo *info;

    Error *err = NULL;



    info = g_malloc(sizeof(*info));

    vnc_init_basic_info_from_server_addr(vd->lsock,

                                         qapi_VncServerInfo_base(info), &err);

    info->has_auth = true;

    info->auth = g_strdup(vnc_auth_name(vd));

    if (err) {

        qapi_free_VncServerInfo(info);

        info = NULL;

        error_free(err);

    }

    return info;

}
