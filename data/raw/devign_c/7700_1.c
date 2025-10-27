int tap_win32_init(VLANState *vlan, const char *model,

                   const char *name, const char *ifname)

{

    TAPState *s;



    s = qemu_mallocz(sizeof(TAPState));

    if (!s)

        return -1;

    if (tap_win32_open(&s->handle, ifname) < 0) {

        printf("tap: Could not open '%s'\n", ifname);

        return -1;

    }



    s->vc = qemu_new_vlan_client(vlan, model, name, tap_receive, NULL, s);



    snprintf(s->vc->info_str, sizeof(s->vc->info_str),

             "tap: ifname=%s", ifname);



    qemu_add_wait_object(s->handle->tap_semaphore, tap_win32_send, s);

    return 0;

}
