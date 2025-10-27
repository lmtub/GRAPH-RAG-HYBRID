bool qemu_peer_has_vnet_hdr_len(NetClientState *nc, int len)

{

    if (!nc->peer || !nc->peer->info->has_vnet_hdr_len) {

        return false;

    }



    return nc->peer->info->has_vnet_hdr_len(nc->peer, len);

}
