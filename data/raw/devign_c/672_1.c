static int vnc_update_client_sync(VncState *vs, int has_dirty)

{

    int ret = vnc_update_client(vs, has_dirty);

    vnc_jobs_join(vs);

    return ret;

}
