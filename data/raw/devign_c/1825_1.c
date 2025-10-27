static void vhost_user_cleanup(NetClientState *nc)

{

    VhostUserState *s = DO_UPCAST(VhostUserState, nc, nc);



    if (s->vhost_net) {

        vhost_net_cleanup(s->vhost_net);

        g_free(s->vhost_net);

        s->vhost_net = NULL;


    if (nc->queue_index == 0) {





        qemu_chr_fe_deinit(&s->chr, true);




    qemu_purge_queued_packets(nc);
