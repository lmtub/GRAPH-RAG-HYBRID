static int net_vhost_user_init(NetClientState *peer, const char *device,

                               const char *name, CharDriverState *chr,

                               int queues)

{

    NetClientState *nc;

    VhostUserState *s;

    int i;



    for (i = 0; i < queues; i++) {

        nc = qemu_new_net_client(&net_vhost_user_info, peer, device, name);



        snprintf(nc->info_str, sizeof(nc->info_str), "vhost-user%d to %s",

                 i, chr->label);



        nc->queue_index = i;



        s = DO_UPCAST(VhostUserState, nc, nc);

        s->chr = chr;

    }



    qemu_chr_add_handlers(chr, NULL, NULL, net_vhost_user_event, (void*)name);



    return 0;

}
