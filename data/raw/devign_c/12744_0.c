void qemu_peer_using_vnet_hdr(NetClientState *nc, bool enable)

{

    if (!nc->peer || !nc->peer->info->using_vnet_hdr) {

        return;

    }



    nc->peer->info->using_vnet_hdr(nc->peer, enable);

}
