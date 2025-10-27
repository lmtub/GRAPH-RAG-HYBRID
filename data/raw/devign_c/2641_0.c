bool qemu_peer_has_ufo(NetClientState *nc)

{

    if (!nc->peer || !nc->peer->info->has_ufo) {

        return false;

    }



    return nc->peer->info->has_ufo(nc->peer);

}
