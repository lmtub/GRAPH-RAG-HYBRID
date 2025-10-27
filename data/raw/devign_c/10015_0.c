void qemu_peer_set_offload(NetClientState *nc, int csum, int tso4, int tso6,

                          int ecn, int ufo)

{

    if (!nc->peer || !nc->peer->info->set_offload) {

        return;

    }



    nc->peer->info->set_offload(nc->peer, csum, tso4, tso6, ecn, ufo);

}
